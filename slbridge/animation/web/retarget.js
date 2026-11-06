import * as THREE from "/animation/vendor/three.module.js";
import { stabilizeQuaternionTrack } from "./stabilization.js";
import { forwardKinematics, solveTwoBone, stabilizeSourcePoses, SOURCE_TEMPORAL_PROFILE } from "./motion-geometry.mjs";
export function axisAngleQuaternion(vector) {
  if (vector?.length !== 3 || !vector.every(Number.isFinite))
    throw new Error("demoInvalidRotation");
  const v = new THREE.Vector3(...vector),
    angle = v.length();
  return angle < 1e-9
    ? new THREE.Quaternion()
    : new THREE.Quaternion().setFromAxisAngle(v.divideScalar(angle), angle);
}
export function sourceWorldRotations(pose, parents) {
  const world = [];
  pose.forEach((v, i) => {
    const q = axisAngleQuaternion(v);
    if (i > 0) q.premultiply(world[parents[i]]);
    world.push(q);
  });
  return world;
}
export function boneFrame(direction, normal) {
  if (
    !direction.toArray().every(Number.isFinite) ||
    direction.lengthSq() < 1e-12 ||
    !normal.toArray().every(Number.isFinite)
  )
    throw new Error("demoInvalidRotation");
  const y = direction.clone().normalize(),
    z = normal.clone().addScaledVector(y, -normal.dot(y));
  if (z.lengthSq() < 1e-8) z.set(0, 1, 0).addScaledVector(y, -y.y);
  if (z.lengthSq() < 1e-8) z.set(1, 0, 0).addScaledVector(y, -y.x);
  z.normalize();
  const x = new THREE.Vector3().crossVectors(y, z).normalize();
  z.crossVectors(x, y);
  return new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().makeBasis(x, y, z),
  );
}
const links = [
  ["neck", 12, 15],
  ["head", 15, null],
  ["leftShoulder", 13, 16],
  ["rightShoulder", 14, 17],
  ["leftUpperArm", 16, 18],
  ["rightUpperArm", 17, 19],
  ["leftLowerArm", 18, 20],
  ["rightLowerArm", 19, 21],
  ["leftHand", 20, 28],
  ["rightHand", 21, 43],
];
for (const [side, starts] of [
  ["left", [37, 25, 28, 34, 31]],
  ["right", [52, 40, 43, 49, 46]],
]) {
  for (const [i, finger] of [
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Little",
  ].entries()) {
    const names =
      i === 0
        ? ["Metacarpal", "Proximal", "Distal"]
        : ["Proximal", "Intermediate", "Distal"];
    for (let j = 0; j < 3; j++)
      links.push([
        side + finger + names[j],
        starts[i] + j,
        j < 2 ? starts[i] + j + 1 : null,
      ]);
  }
}
export function createT08Retarget(rig, modelRoot, bundle) {
  const playingPose = rig.capture();
  let binding, sourceBasis, targetBasis, armSettings, sourceCenter, targetCenter;
  let diagnostics = null;
  const position = name => rig.bones[name].getWorldPosition(new THREE.Vector3());
  function basis(left, right, up) {
    const x = left.clone().sub(right).normalize();
    const y = up.clone().addScaledVector(x, -up.dot(x)).normalize();
    const z = new THREE.Vector3().crossVectors(x, y).normalize();
    return new THREE.Matrix3().set(x.x,y.x,z.x, x.y,y.y,z.y, x.z,y.z,z.z);
  }
  try {
    rig.reset();
    modelRoot.updateMatrixWorld(true);
    const rest = bundle.restJoints.map((v) => new THREE.Vector3(...v));
    sourceBasis = basis(rest[16], rest[17], rest[12].clone().sub(rest[0])).transpose();
    targetBasis = basis(position("leftUpperArm"), position("rightUpperArm"), position("neck").sub(position("hips")));
    armSettings = [["left",16,18,20], ["right",17,19,21]].map(([side,upper,lower,hand]) => ({
      side, upper, lower, hand,
      scale: (position(side+"UpperArm").distanceTo(position(side+"LowerArm")) +
        position(side+"LowerArm").distanceTo(position(side+"Hand"))) /
        (rest[upper].distanceTo(rest[lower]) + rest[lower].distanceTo(rest[hand])),
    }));
    sourceCenter = rest[16].clone().add(rest[17]).multiplyScalar(0.5);
    targetCenter = position("leftUpperArm").add(position("rightUpperArm")).multiplyScalar(0.5);
    const palmNormals = {};
    for (const [side, w, index, little] of [
      ["left", 20, 25, 31],
      ["right", 21, 40, 46],
    ])
      palmNormals[side] = new THREE.Vector3()
        .crossVectors(
          rest[index].clone().sub(rest[w]),
          rest[little].clone().sub(rest[w]),
        )
        .normalize();
    const targetPalm = {};
    for (const side of ["left", "right"]) {
      const w = rig.bones[side + "Hand"].getWorldPosition(new THREE.Vector3());
      const index = rig.bones[side + "IndexProximal"]
        .getWorldPosition(new THREE.Vector3())
        .sub(w);
      const little = rig.bones[side + "LittleProximal"]
        .getWorldPosition(new THREE.Vector3())
        .sub(w);
      targetPalm[side] = new THREE.Vector3()
        .crossVectors(index, little)
        .normalize();
    }
    binding = links.map(([name, joint, next]) => {
      const bone = rig.bones[name];
      if (!bone) throw new Error("demoMissingBone");
      const restWorld = bone.getWorldQuaternion(new THREE.Quaternion());
      const restPosition = bone.getWorldPosition(new THREE.Vector3());
      const sourceDir =
        next !== null
          ? rest[next].clone().sub(rest[joint])
          : rest[joint].clone().sub(rest[bundle.parents[joint]]);
      let targetChild;
      if (name.endsWith("UpperArm"))
        targetChild = rig.bones[name.replace("UpperArm", "LowerArm")];
      else if (name.endsWith("LowerArm"))
        targetChild = rig.bones[name.replace("LowerArm", "Hand")];
      else if (name.endsWith("Shoulder"))
        targetChild = rig.bones[name.replace("Shoulder", "UpperArm")];
      else if (name.endsWith("Hand"))
        targetChild = rig.bones[name.replace("Hand", "MiddleProximal")];
      else if (name === "neck") targetChild = rig.bones.head;
      else if (next !== null) targetChild = bone.children.find((c) => c.isBone);
      const targetDir = targetChild
        ? targetChild.getWorldPosition(new THREE.Vector3()).sub(restPosition)
        : restPosition
            .clone()
            .sub(bone.parent.getWorldPosition(new THREE.Vector3()));
      const side = name.startsWith("left")
        ? "left"
        : name.startsWith("right")
          ? "right"
          : null;
      const isFinger = joint >= 25 || name.endsWith("Hand");
      const sourceNormal = isFinger
        ? palmNormals[side]
        : new THREE.Vector3(0, 0, 1);
      const targetNormal = isFinger
        ? targetPalm[side]
        : new THREE.Vector3(0, 0, 1);
      const correction = boneFrame(sourceDir, sourceNormal).multiply(
        boneFrame(targetDir, targetNormal).invert(),
      );
      // Neck/head are already upright in both skeletons; face/hair child offsets
      // are not segment directions and must not rotate the avatar's head at REST.
      const alignedRest =
        name === "neck" || name === "head"
          ? restWorld
          : correction.multiply(restWorld);
      if (rig.profile.corrections?.[name]) alignedRest.multiply(axisAngleQuaternion(rig.profile.corrections[name]));
      return { name, bone, joint, alignedRest };
    });
  } finally {
    rig.interpolate(playingPose, playingPose, 1);
    modelRoot.updateMatrixWorld(true);
  }
  function poseTargets(pose) {
    rig.reset();
    modelRoot.updateMatrixWorld(true);
    const world = [];
    pose.forEach((v, i) => {
      const q = axisAngleQuaternion(v);
      if (i > 0) q.premultiply(world[bundle.parents[i]]);
      world.push(q);
    });
    const targets = {};
    for (const b of binding) {
      const desired = world[b.joint].clone().multiply(b.alignedRest);
      const parentWorld = b.bone.parent.getWorldQuaternion(
        new THREE.Quaternion(),
      );
      b.bone.quaternion.copy(parentWorld.invert().multiply(desired));
      b.bone.updateWorldMatrix(false, true);
      targets[b.name] = b.bone.quaternion.clone();
    }
    // Jaw values belong to SMPL-X, whose axes differ; hold avatar face at rest for this prototype.
    rig.reset();
    modelRoot.updateMatrixWorld(true);
    return targets;
  }
  function setWorldRotation(bone, world) {
    const parent = bone.parent.getWorldQuaternion(new THREE.Quaternion());
    bone.quaternion.copy(parent.invert().multiply(world));
    bone.updateWorldMatrix(false, true);
  }
  function pointBone(bone, child, target) {
    const origin = bone.getWorldPosition(new THREE.Vector3());
    const from = child.getWorldPosition(new THREE.Vector3()).sub(origin).normalize();
    const to = target.clone().sub(origin).normalize();
    const delta = new THREE.Quaternion().setFromUnitVectors(from, to);
    setWorldRotation(bone, delta.multiply(bone.getWorldQuaternion(new THREE.Quaternion())));
  }
  function constrain(frame, pose) {
    rig.interpolate(frame, frame, 1); modelRoot.updateMatrixWorld(true);
    const source = forwardKinematics(pose, bundle.parents, bundle.restJoints).positions;
    for (const arm of armSettings) {
      const upper = rig.bones[arm.side+"UpperArm"], lower = rig.bones[arm.side+"LowerArm"], hand = rig.bones[arm.side+"Hand"];
      const root = position(arm.side+"UpperArm");
      const offset = new THREE.Vector3(...source[arm.hand]).sub(new THREE.Vector3(...source[arm.upper]));
      const goal = offset.applyMatrix3(sourceBasis).applyMatrix3(targetBasis).multiplyScalar(arm.scale).add(root);
      const before = position(arm.side+"Hand").distanceTo(goal);
      const orientation = hand.getWorldQuaternion(new THREE.Quaternion());
      const solved = solveTwoBone(root.toArray(), position(arm.side+"LowerArm").toArray(), position(arm.side+"Hand").toArray(), goal.toArray());
      pointBone(upper, lower, new THREE.Vector3(...solved.elbow));
      pointBone(lower, hand, new THREE.Vector3(...solved.wrist));
      setWorldRotation(hand, orientation);
      diagnostics.samples++;
      diagnostics.before_sum += before;
      diagnostics.after_sum += position(arm.side+"Hand").distanceTo(goal);
      diagnostics.clamped += Number(solved.clamped);
    }
    return Object.fromEntries(binding.map(b => [b.name, b.bone.quaternion.clone()]));
  }
  function compile(
    clip,
    {
      stabilize = clip.experimental === true,
      stabilizationProfile = clip.display_stabilization_profile,
      positionConstraints = false,
    } = {},
  ) {
    const snapshot = rig.capture();
    try {
      const sourceSpace = stabilize && stabilizationProfile === SOURCE_TEMPORAL_PROFILE;
      const poses = sourceSpace ? stabilizeSourcePoses(clip.poses,clip.fps) : clip.poses;
      const raw = poses.map((p) => poseTargets(p));
      const frames = stabilize && !sourceSpace
        ? stabilizeQuaternionTrack(raw, clip.fps, stabilizationProfile)
        : raw;
      diagnostics = { strategy: "source-wrist-ik-v1", samples: 0, before_sum: 0, after_sum: 0, clamped: 0 };
      return positionConstraints ? frames.map((frame, i) => constrain(frame, poses[i])) : frames;
    } finally {
      rig.interpolate(snapshot, snapshot, 1);
      modelRoot.updateMatrixWorld(true);
    }
  }
  return { compile, mappedBones: binding.length,
    sourcePositions(pose) {
      const scale = (armSettings[0].scale + armSettings[1].scale)/2;
      return forwardKinematics(pose,bundle.parents,bundle.restJoints).positions.map(p =>
        new THREE.Vector3(...p).sub(sourceCenter).applyMatrix3(sourceBasis).applyMatrix3(targetBasis).multiplyScalar(scale).add(targetCenter));
    },
    get diagnostics() {
    if (!diagnostics?.samples) return null;
    return { strategy: diagnostics.strategy, samples: diagnostics.samples, clamped: diagnostics.clamped,
      wrist_error_before_mean: diagnostics.before_sum / diagnostics.samples,
      wrist_error_after_mean: diagnostics.after_sum / diagnostics.samples };
  } };
}
