"""The experiment-2 double pendulum in Isaac Sim (PhysX), same inertial specification.

Link 1 is PD-driven; link 2 hangs from its end, hinged 40% along its own length, and is free.
"""

import numpy as np

M1 = M2 = 0.45359237  # 1 lb
L1 = L2 = 0.3048  # 1 ft
HINGE_FRAC = 0.40
G = 9.81
KP0, KD0 = 80.0, 2.0
F_CTRL = 50

ARMS = ("A", "B", "C")
ARM_LABEL = {
    "A": "A: TGS + explicit PD",
    "B": "B: TGS + PhysX joint drive",
    "C": "C: PGS + PhysX joint drive",
}


def build_scene(n_envs, arm, spacing=1.0):
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
    px.CreateEnableGPUDynamicsAttr().Set(False)
    px.CreateEnableStabilizationAttr().Set(False)

    a, b = HINGE_FRAC * L2, (1 - HINGE_FRAC) * L2
    for i in range(n_envs):
        root = f"/World/env{i}"
        x0 = i * spacing
        UsdGeom.Xform.Define(stage, root)
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root))
        PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root))

        base = f"{root}/base"
        bx = UsdGeom.Xform.Define(stage, base)
        bx.AddTranslateOp().Set(Gf.Vec3d(x0, 0.0, 0.0))
        bprim = stage.GetPrimAtPath(base)
        UsdPhysics.RigidBodyAPI.Apply(bprim)
        bm = UsdPhysics.MassAPI.Apply(bprim)
        bm.CreateMassAttr().Set(1e-6)
        bm.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1e-9, 1e-9, 1e-9))
        anchor = UsdPhysics.FixedJoint.Define(stage, f"{root}/anchor")
        anchor.CreateBody0Rel().SetTargets([])
        anchor.CreateBody1Rel().SetTargets([base])
        anchor.CreateLocalPos0Attr().Set(Gf.Vec3f(x0, 0.0, 0.0))

        def link(path, z, mass, com_z, ixx, geom_from, geom_to, radius):
            xf = UsdGeom.Xform.Define(stage, path)
            xf.AddTranslateOp().Set(Gf.Vec3d(x0, 0.0, z))
            prim = stage.GetPrimAtPath(path)
            UsdPhysics.RigidBodyAPI.Apply(prim)
            m = UsdPhysics.MassAPI.Apply(prim)
            m.CreateMassAttr().Set(mass)
            m.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, com_z))
            m.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(ixx, ixx, 1e-9))
            m.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            cap = UsdGeom.Capsule.Define(stage, f"{path}/geom")
            cap.CreateAxisAttr().Set("Z")
            cap.CreateHeightAttr().Set(abs(geom_to - geom_from))
            cap.CreateRadiusAttr().Set(radius)
            cap.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, (geom_from + geom_to) / 2))
            cap.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

        link(f"{root}/link1", 0.0, M1, -L1 / 2, M1 * L1**2 / 12, 0.0, -L1, 0.009)
        link(f"{root}/link2", -L1, M2, (a - b) / 2, M2 * L2**2 / 12, a, -b, 0.007)

        hinge1 = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/j1")
        hinge1.CreateBody0Rel().SetTargets([base])
        hinge1.CreateBody1Rel().SetTargets([f"{root}/link1"])
        hinge1.CreateAxisAttr().Set("Y")
        hinge1.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        hinge1.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        jp = hinge1.GetPrim()
        PhysxSchema.PhysxJointAPI.Apply(jp)
        drive = UsdPhysics.DriveAPI.Apply(jp, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateMaxForceAttr().Set(1e6)
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)
        drive.CreateTargetPositionAttr().Set(0.0)

        hinge2 = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/j2")
        hinge2.CreateBody0Rel().SetTargets([f"{root}/link1"])
        hinge2.CreateBody1Rel().SetTargets([f"{root}/link2"])
        hinge2.CreateAxisAttr().Set("Y")
        hinge2.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -L1))
        hinge2.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        PhysxSchema.PhysxJointAPI.Apply(hinge2.GetPrim())

    return Articulation(prim_paths_expr="/World/env.*", name="doubles")


class IsaacDouble:
    """Steps n_envs copies of the double pendulum, one parameter set per environment."""

    def __init__(self, world, arm, n_envs):
        self.world, self.arm, self.n = world, arm, n_envs
        self.view = build_scene(n_envs, arm)
        world.scene.add(self.view)
        world.reset()
        self.view.initialize()
        self.driven = list(self.view.dof_names).index("j1")

    def set_dt(self, h):
        self.world.set_simulation_dt(physics_dt=h, rendering_dt=h)

    def _apply(self, params):
        p = np.atleast_2d(np.asarray(params, dtype=np.float32))
        if len(p) < self.n:
            p = np.concatenate([p, np.repeat(p[:1], self.n - len(p), axis=0)])
        arm_col = np.zeros((self.n, 2), np.float32)
        arm_col[:, self.driven] = p[:, 0]
        self.view.set_armatures(arm_col)
        kps = np.zeros((self.n, 2), np.float32)
        kds = np.zeros((self.n, 2), np.float32)
        if self.arm != "A":  # joint 2 is always free: no drive on it
            kps[:, self.driven] = p[:, 1]
            kds[:, self.driven] = p[:, 2]
        self.view.set_gains(kps=kps, kds=kds)
        return p

    def rollout(self, refs, dec, params, f_ctrl=F_CTRL):
        """Both joint angles after each control step, shape (rows, len(refs), 2)."""
        self.world.reset()
        rows = len(np.atleast_2d(params))
        p = self._apply(params)
        zeros = np.zeros((self.n, 2), dtype=np.float32)
        self.view.set_joint_positions(zeros)
        self.view.set_joint_velocities(zeros)
        out = np.empty((self.n, len(refs), 2))
        other = 1 - self.driven
        for k, r in enumerate(refs):
            for _ in range(dec):
                if self.arm == "A":
                    q = self.view.get_joint_positions()
                    v = self.view.get_joint_velocities()
                    tau = np.zeros((self.n, 2), np.float32)
                    tau[:, self.driven] = (p[:, 1] * (r - q[:, self.driven])
                                           - p[:, 2] * v[:, self.driven])
                    self.view.set_joint_efforts(tau)
                else:
                    tgt = self.view.get_joint_positions().astype(np.float32)
                    tgt[:, self.driven] = r
                    tgt[:, other] = 0.0
                    self.view.set_joint_position_targets(tgt)
                self.world.step(render=False)
            out[:, k] = self.view.get_joint_positions()
        order = [self.driven, other]
        return out[:rows][..., order]  # columns ordered as (driven, free)
