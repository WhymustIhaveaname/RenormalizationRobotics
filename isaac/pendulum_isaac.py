"""The experiment-1 pendulum in Isaac Sim (PhysX), built to the same inertial specification.

Run inside the Isaac Sim venv; `SimulationApp` must be constructed before any other import.
"""

import numpy as np

# Same uniform rod and gains as the MuJoCo experiments (SI units).
M = 0.45359237  # 1 lb
L = 0.3048  # 1 ft
G = 9.81
J0 = M * L**2 / 3
KP0, KD0 = 80.0, 2.0
F_CTRL = 50

ARMS = ("A", "B", "C")
ARM_LABEL = {
    "A": "A: TGS + explicit PD",
    "B": "B: TGS + PhysX joint drive",
    "C": "C: PGS + PhysX joint drive",
}


def build_scene(n_envs, arm, spacing=1.0):
    """One pendulum per environment, all identical; returns the articulation view."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    import omni.usd
    from isaacsim.core.prims import Articulation

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(G)
    px = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    px.CreateSolverTypeAttr().Set("PGS" if arm == "C" else "TGS")
    px.CreateEnableGPUDynamicsAttr().Set(False)  # 1-2 DOF; the CPU pipeline is faster here
    px.CreateEnableStabilizationAttr().Set(False)

    for i in range(n_envs):
        root = f"/World/env{i}"
        UsdGeom.Xform.Define(stage, root)
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root))
        PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root))

        # PhysX needs a base link fixed to the world; the rod then hinges off it.
        base = f"{root}/base"
        bx = UsdGeom.Xform.Define(stage, base)
        bx.AddTranslateOp().Set(Gf.Vec3d(i * spacing, 0.0, 0.0))
        bprim = stage.GetPrimAtPath(base)
        UsdPhysics.RigidBodyAPI.Apply(bprim)
        bmass = UsdPhysics.MassAPI.Apply(bprim)
        bmass.CreateMassAttr().Set(1e-6)
        bmass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1e-9, 1e-9, 1e-9))
        anchor = UsdPhysics.FixedJoint.Define(stage, f"{root}/anchor")
        anchor.CreateBody0Rel().SetTargets([])
        anchor.CreateBody1Rel().SetTargets([base])
        anchor.CreateLocalPos0Attr().Set(Gf.Vec3f(i * spacing, 0.0, 0.0))
        anchor.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        link = f"{root}/link"
        xform = UsdGeom.Xform.Define(stage, link)
        xform.AddTranslateOp().Set(Gf.Vec3d(i * spacing, 0.0, 0.0))
        prim = stage.GetPrimAtPath(link)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr().Set(M)
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, -L / 2))
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(M * L**2 / 12, M * L**2 / 12, 1e-9))
        mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        rod = UsdGeom.Capsule.Define(stage, f"{link}/geom")
        rod.CreateAxisAttr().Set("Z")
        rod.CreateHeightAttr().Set(L)
        rod.CreateRadiusAttr().Set(0.008)
        rod.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -L / 2))
        rod.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)  # visual only, no collision

        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/hinge")
        joint.CreateBody0Rel().SetTargets([base])
        joint.CreateBody1Rel().SetTargets([link])
        joint.CreateAxisAttr().Set("Y")
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        jp = joint.GetPrim()
        PhysxSchema.PhysxJointAPI.Apply(jp)
        drive = UsdPhysics.DriveAPI.Apply(jp, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateMaxForceAttr().Set(1e6)
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)
        drive.CreateTargetPositionAttr().Set(0.0)

    return Articulation(prim_paths_expr="/World/env.*", name="pendulums")


class IsaacPendulum:
    """Steps n_envs copies of the pendulum, one parameter set per environment."""

    def __init__(self, world, arm, n_envs):
        self.world, self.arm, self.n = world, arm, n_envs
        self.view = build_scene(n_envs, arm)
        world.scene.add(self.view)
        world.reset()
        self.view.initialize()

    def _apply(self, params):
        """Write the parameters; must run after world.reset(), which restores USD values.

        Fewer rows than environments is allowed: the rest are padded with the first row and
        their output is discarded, so a short batch costs no more than a full one.
        """
        p = np.atleast_2d(np.asarray(params, dtype=np.float32))
        if len(p) < self.n:
            p = np.concatenate([p, np.repeat(p[:1], self.n - len(p), axis=0)])
        self.view.set_armatures(p[:, :1])
        if self.arm == "A":  # torques come from Python, so the PhysX drive stays off
            z = np.zeros((self.n, 1), np.float32)
            self.view.set_gains(kps=z, kds=z)
        else:
            self.view.set_gains(kps=p[:, 1:2], kds=p[:, 2:3])
        return p

    def set_dt(self, h):
        self.world.set_simulation_dt(physics_dt=h, rendering_dt=h)

    def rollout(self, refs, dec, params, f_ctrl=F_CTRL):
        """Joint angle after each control step, shape (n_envs, len(refs)).

        `params` is (n_envs, 3) of (armature, K_P, K_D); each environment runs its own set.
        """
        self.world.reset()
        k = len(np.atleast_2d(params))
        p = self._apply(params)
        zeros = np.zeros((self.n, 1), dtype=np.float32)
        self.view.set_joint_positions(zeros)
        self.view.set_joint_velocities(zeros)
        k_rows = k
        out = np.empty((self.n, len(refs)))
        for k, r in enumerate(refs):
            target = np.full((self.n, 1), r, dtype=np.float32)
            for _ in range(dec):
                if self.arm == "A":
                    q = self.view.get_joint_positions()
                    v = self.view.get_joint_velocities()
                    self.view.set_joint_efforts(
                        (p[:, 1:2] * (target - q) - p[:, 2:3] * v).astype(np.float32))
                else:
                    self.view.set_joint_position_targets(target)
                self.world.step(render=False)
            out[:, k] = self.view.get_joint_positions().reshape(self.n)
        return out[:k_rows]
