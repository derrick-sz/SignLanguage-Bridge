import { validateRigProfile } from "./expression.mjs";
import { JAW_AVATAR_SHA256, MAX_JAW_ANGLE } from "./nonmanual.mjs";

const GODETTE_V11_SHA256 = "e4a5766d98ab53871415da3ef4609c4848386c8b4de2958d5e420a8d7524d476";

const GODETTE_V12_SHA256 = "a7942f7a69e8fb078d50767e85944c0e59b47dc20b40479ab86d8c16513717c4";

const GODETTE_V13_SHA256 = "2d89e2e33271d2e06c5a834a56675663aa5551814e2c98ef86fc084ca0c151a8";

const HUMANOID_BONES = Object.freeze([
  "root",
  "body",
  "hips",
  "spine",
  "chest",
  "upperChest",
  "neck",
  "neck2",
  "neck3",
  "head",
  "rightShoulder",
  "rightUpperArm",
  "rightLowerArm",
  "rightHand",
  "leftShoulder",
  "leftUpperArm",
  "leftLowerArm",
  "leftHand",
  "rightThumbMetacarpal",
  "rightThumbProximal",
  "rightThumbDistal",
  "rightIndexProximal",
  "rightIndexIntermediate",
  "rightIndexDistal",
  "rightMiddleProximal",
  "rightMiddleIntermediate",
  "rightMiddleDistal",
  "rightRingProximal",
  "rightRingIntermediate",
  "rightRingDistal",
  "rightLittleProximal",
  "rightLittleIntermediate",
  "rightLittleDistal",
  "leftThumbMetacarpal",
  "leftThumbProximal",
  "leftThumbDistal",
  "leftIndexProximal",
  "leftIndexIntermediate",
  "leftIndexDistal",
  "leftMiddleProximal",
  "leftMiddleIntermediate",
  "leftMiddleDistal",
  "leftRingProximal",
  "leftRingIntermediate",
  "leftRingDistal",
  "leftLittleProximal",
  "leftLittleIntermediate",
  "leftLittleDistal",
  "jaw",
  "leftUpperEyelid",
  "rightUpperEyelid",
  "leftLowerEyelid",
  "rightLowerEyelid",
]);

const REQUIRED_SIGNING_BONES = Object.freeze([
  "rightUpperArm",
  "rightLowerArm",
  "rightHand",
  "leftUpperArm",
  "leftLowerArm",
  "leftHand",
  "rightThumbMetacarpal",
  "rightThumbProximal",
  "rightThumbDistal",
  "rightIndexProximal",
  "rightIndexIntermediate",
  "rightIndexDistal",
  "rightMiddleProximal",
  "rightMiddleIntermediate",
  "rightMiddleDistal",
  "rightRingProximal",
  "rightRingIntermediate",
  "rightRingDistal",
  "rightLittleProximal",
  "rightLittleIntermediate",
  "rightLittleDistal",
  "leftThumbMetacarpal",
  "leftThumbProximal",
  "leftThumbDistal",
  "leftIndexProximal",
  "leftIndexIntermediate",
  "leftIndexDistal",
  "leftMiddleProximal",
  "leftMiddleIntermediate",
  "leftMiddleDistal",
  "leftRingProximal",
  "leftRingIntermediate",
  "leftRingDistal",
  "leftLittleProximal",
  "leftLittleIntermediate",
  "leftLittleDistal",
]);

const GODETTE_PROFILE = Object.freeze({
  id: "godette-v1",
  motionSpace: "humanoid-local-v1",
  bones: Object.freeze({
    root: ["Root_225"],
    body: ["Body_220"],
    hips: ["Hip_218"],
    spine: ["Spine_1_199"],
    chest: ["Spine_2_198"],
    upperChest: ["Ribcage_197"],
    neck: ["Neck_1_132"],
    neck2: ["Neck_2_131"],
    neck3: ["Neck_3_130"],
    head: ["Head_129"],
    rightShoulder: ["ClavicR_192", "Clavic.R_192"],
    rightUpperArm: ["Arm_Upper_1R_187", "Arm_Upper_1.R_187"],
    rightLowerArm: ["Arm_Lower_1R_185", "Arm_Lower_1.R_185"],
    rightHand: ["HandR_184", "Hand.R_184"],
    leftShoulder: ["ClavicL_162", "Clavic.L_162"],
    leftUpperArm: ["Arm_Upper_1L_157", "Arm_Upper_1.L_157"],
    leftLowerArm: ["Arm_Lower_1L_155", "Arm_Lower_1.L_155"],
    leftHand: ["HandL_154", "Hand.L_154"],
    rightThumbMetacarpal: ["ThumbR_178", "Thumb.R_178"],
    rightThumbProximal: ["ThumbR002_177", "Thumb.R.002_177"],
    rightThumbDistal: ["ThumbR001_176", "Thumb.R.001_176"],
    rightIndexProximal: ["Finger_1R_166", "Finger_1.R_166"],
    rightIndexIntermediate: ["Finger_1R002_165", "Finger_1.R.002_165"],
    rightIndexDistal: ["Finger_1R001_164", "Finger_1.R.001_164"],
    rightMiddleProximal: ["Finger_2R_169", "Finger_2.R_169"],
    rightMiddleIntermediate: ["Finger_2R002_168", "Finger_2.R.002_168"],
    rightMiddleDistal: ["Finger_2R001_167", "Finger_2.R.001_167"],
    rightRingProximal: ["Finger_3R_172", "Finger_3.R_172"],
    rightRingIntermediate: ["Finger_3R002_171", "Finger_3.R.002_171"],
    rightRingDistal: ["Finger_3R001_170", "Finger_3.R.001_170"],
    rightLittleProximal: ["Finger_4R_175", "Finger_4.R_175"],
    rightLittleIntermediate: ["Finger_4R002_174", "Finger_4.R.002_174"],
    rightLittleDistal: ["Finger_4R001_173", "Finger_4.R.001_173"],
    leftThumbMetacarpal: ["ThumbL_148", "Thumb.L_148"],
    leftThumbProximal: ["ThumbL002_147", "Thumb.L.002_147"],
    leftThumbDistal: ["ThumbL001_146", "Thumb.L.001_146"],
    leftIndexProximal: ["Finger_1L_136", "Finger_1.L_136"],
    leftIndexIntermediate: ["Finger_1L002_135", "Finger_1.L.002_135"],
    leftIndexDistal: ["Finger_1L001_134", "Finger_1.L.001_134"],
    leftMiddleProximal: ["Finger_2L_139", "Finger_2.L_139"],
    leftMiddleIntermediate: ["Finger_2L002_138", "Finger_2.L.002_138"],
    leftMiddleDistal: ["Finger_2L001_137", "Finger_2.L.001_137"],
    leftRingProximal: ["Finger_3L_142", "Finger_3.L_142"],
    leftRingIntermediate: ["Finger_3L002_141", "Finger_3.L.002_141"],
    leftRingDistal: ["Finger_3L001_140", "Finger_3.L.001_140"],
    leftLittleProximal: ["Finger_4L_145", "Finger_4.L_145"],
    leftLittleIntermediate: ["Finger_4L002_144", "Finger_4.L.002_144"],
    leftLittleDistal: ["Finger_4L001_143", "Finger_4.L.001_143"],
    jaw: ["Jaw_4"],
    leftUpperEyelid: ["Eyelid_Control_UpperL_14"],
    rightUpperEyelid: ["Eyelid_Control_UpperR_15"],
    leftLowerEyelid: ["Eyelid_Control_LowerL_16"],
    rightLowerEyelid: ["Eyelid_Control_LowerR_17"],
  }),
});

const LEGACY_BONE_ALIASES = Object.freeze({
  hip: "hips",
  ribcage: "upperChest",
  rShoulder: "rightShoulder",
  rUpperArm: "rightUpperArm",
  rForearm: "rightLowerArm",
  rHand: "rightHand",
  lShoulder: "leftShoulder",
  lUpperArm: "leftUpperArm",
  lForearm: "leftLowerArm",
  lHand: "leftHand",
  rthumb: "rightThumbMetacarpal",
  rindex: "rightIndexIntermediate",
  rmiddle: "rightMiddleIntermediate",
  rring: "rightRingIntermediate",
  rpinky: "rightLittleIntermediate",
  lthumb: "leftThumbMetacarpal",
  lindex: "leftIndexIntermediate",
  lmiddle: "leftMiddleIntermediate",
  lring: "leftRingIntermediate",
  lpinky: "leftLittleIntermediate",
  lUpperEyelid: "leftUpperEyelid",
  rUpperEyelid: "rightUpperEyelid",
  lLowerEyelid: "leftLowerEyelid",
  rLowerEyelid: "rightLowerEyelid",
});

function normalizeBoneName(name) {
  return String(name || "")
    .replace(/[._\s-]/g, "")
    .toLowerCase();
}

function canonicalizePose(pose = {}) {
  const canonical = {};
  for (const [name, rotation] of Object.entries(pose)) {
    canonical[LEGACY_BONE_ALIASES[name] || name] = rotation;
  }
  return canonical;
}

function indexSceneBones(root) {
  const exact = new Map();
  const normalized = new Map();
  const skeletons = new Set();
  root?.traverse?.((object) => {
    if (object.isBone) {
      exact.set(object.name, object);
      normalized.set(normalizeBoneName(object.name), object);
    }
    if (object.isSkinnedMesh && object.skeleton) skeletons.add(object.skeleton);
  });
  return { exact, normalized, skeletons: [...skeletons] };
}

function resolveProfileBones(root, profile = GODETTE_PROFILE) {
  const index = indexSceneBones(root);
  const bones = {};
  for (const [canonicalName, candidates] of Object.entries(profile.bones)) {
    for (const candidate of candidates) {
      const bone =
        index.exact.get(candidate) ||
        index.normalized.get(normalizeBoneName(candidate));
      if (bone) {
        bones[canonicalName] = bone;
        break;
      }
    }
  }
  return { bones, skeletons: index.skeletons };
}

function createHumanoidRig(root, profile = GODETTE_PROFILE, {avatarSha256=null} = {}) {
  validateRigProfile(profile);
  const { bones, skeletons } = resolveProfileBones(root, profile);
  const rest = {};
  for (const [name, bone] of Object.entries(bones)) {
    rest[name] = {
      quaternion: bone.quaternion.clone(),
      position: bone.position.clone(),
      rotation: {
        x: bone.rotation.x,
        y: bone.rotation.y,
        z: bone.rotation.z,
        order: bone.rotation.order,
      },
    };
  }

  // Controls are selected from deforming bones; editor-only eyelid controls have
  // zero skin weights in Godette and do not move any vertices after GLB export.
  root.updateMatrixWorld(true);
  const weighted = new Map();
  root.traverse(object => {
    if (!object.isSkinnedMesh) return;
    const attrs = object.geometry.attributes;
    if (!attrs.skinIndex || !attrs.skinWeight || !attrs.position) return;
    for (let i=0; i<attrs.position.count; i++) {
      const p = object.position.clone().fromBufferAttribute(attrs.position,i).applyMatrix4(object.matrixWorld);
      if (bones.head) bones.head.worldToLocal(p);
      for (let k=0;k<4;k++) {
        const weight = attrs.skinWeight.getComponent(i,k), bone = object.skeleton.bones[attrs.skinIndex.getComponent(i,k)];
        if (weight <= 0.001 || !/^eyelid[lr]\d/.test(normalizeBoneName(bone?.name))) continue;
        const item=weighted.get(bone)||{sum:0,weight:0}; item.sum+=p.y*weight; item.weight+=weight; weighted.set(bone,item);
      }
    }
  });
  const eyeCenter = weighted.size ? [...weighted.values()].reduce((s,x)=>s+x.sum/x.weight,0)/weighted.size : 0;
  const faceControls = [], facialRest = new Map();
  root.traverse(bone => {
    if (!bone.isBone) return;
    const name = normalizeBoneName(bone.name);
    const brow = /^eyebrow[lr]\d/.test(name), eyelid = weighted.get(bone);
    if (!brow && !eyelid && bone !== bones.jaw) return;
    facialRest.set(bone,{position:bone.position.clone(),quaternion:bone.quaternion.clone()});
    if (!brow && !eyelid) return;
    const offset = bone.position.clone().set(0, brow ? 0.006 : (eyelid.sum/eyelid.weight > eyeCenter ? -0.014 : 0.009), 0);
    if (bones.head) offset.applyQuaternion(bones.head.getWorldQuaternion(bone.quaternion.clone()));
    offset.applyQuaternion(bone.parent.getWorldQuaternion(bone.quaternion.clone()).invert());
    faceControls.push({bone,kind:brow?'brow':'eyelid',offset});
  });
  // The exported lower-lip/teeth group is a sibling of Jaw_4, not its child.
  // Only the measured avatar may use this calibration; names alone prove nothing.
  const lowerLipGroup=root.getObjectByName?.('Lips_Lower_13');
  const jawCalibrationAvailable=Boolean(avatarSha256===JAW_AVATAR_SHA256 && bones.jaw?.name==='Jaw_4' &&
    lowerLipGroup?.isBone && lowerLipGroup.parent===bones.jaw.parent);
  if(jawCalibrationAvailable)facialRest.set(lowerLipGroup,{position:lowerLipGroup.position.clone(),quaternion:lowerLipGroup.quaternion.clone()});
  // Frozen v11 and neutral v12/v13 have a separate manual-only mouth driver.
  // v12/v13 must remain closed if its controls fail validation, never jaw-only.
  // Never reuse the legacy
  // source-jaw calibration or enable it from matching bone names alone.
  const mouthCorners=["Lip_L007_8", "Lip_R007_11"].map(name=>root.getObjectByName?.(name));
  const manualMouthRequired=avatarSha256===GODETTE_V12_SHA256 || avatarSha256===GODETTE_V13_SHA256;
  const manualMouthAvailable=Boolean((avatarSha256===GODETTE_V11_SHA256 || manualMouthRequired) && profile.id==='godette-v1' &&
    bones.jaw?.name==='Jaw_4' && lowerLipGroup?.isBone && lowerLipGroup.parent===bones.jaw.parent &&
    mouthCorners.every(bone=>bone?.isBone && bone.parent===lowerLipGroup));
  const manualJawMaxAngle=manualMouthAvailable?0.04:manualMouthRequired?0:MAX_JAW_ANGLE;
  if(manualMouthAvailable){
    for(const bone of [lowerLipGroup,...mouthCorners])
      facialRest.set(bone,{position:bone.position.clone(),quaternion:bone.quaternion.clone()});
  }
  function applyManualMouth(amount){
    if(!Number.isFinite(amount) || amount<0 || amount>1)throw new Error('Invalid manual mouth intensity');
    if(!manualMouthAvailable)return false;
    for(const bone of [bones.jaw,lowerLipGroup])bone.rotateX(amount*manualJawMaxAngle);
    // Counter-rotate seam bones in the lower-lip parent's coordinates.
    for(const bone of mouthCorners){
      const axis=bone.position.clone().set(1,0,0);
      const counter=bone.quaternion.clone().setFromAxisAngle(axis,-amount*manualJawMaxAngle);
      bone.position.applyQuaternion(counter);bone.quaternion.premultiply(counter);
    }
    return true;
  }
  function resetExpression() {
    for (const [bone,base] of facialRest) {bone.position.copy(base.position);bone.quaternion.copy(base.quaternion);}
  }
  function applyCalibratedJaw(angle) {
    if(!Number.isFinite(angle) || angle<0 || angle>MAX_JAW_ANGLE)throw new Error('Invalid calibrated jaw angle');
    if(!jawCalibrationAvailable)return false;
    const base=rest.jaw,lip=facialRest.get(lowerLipGroup);
    const axis=base.position.clone().set(1,0,0);
    const delta=base.quaternion.clone().setFromAxisAngle(axis,angle);
    const parentBase=base.quaternion.clone().normalize();
    const parentDelta=parentBase.clone().multiply(delta).multiply(parentBase.clone().invert());
    bones.jaw.quaternion.copy(base.quaternion).multiply(delta);
    lowerLipGroup.position.copy(lip.position).sub(base.position).applyQuaternion(parentDelta).add(base.position);
    lowerLipGroup.quaternion.copy(parentDelta).multiply(lip.quaternion);
    return true;
  }

  function update() {
    for (const skeleton of skeletons) skeleton.update();
  }

  function createTargets(pose = {}) {
    const canonicalPose = canonicalizePose(pose);
    const targets = {};
    for (const [name, bone] of Object.entries(bones)) {
      const base = rest[name];
      const delta = canonicalPose[name];
      if (!delta) {
        targets[name] = base.quaternion.clone();
        continue;
      }
      const rotation = bone.rotation.clone();
      rotation.set(
        base.rotation.x + (delta.x || 0),
        base.rotation.y + (delta.y || 0),
        base.rotation.z + (delta.z || 0),
        base.rotation.order,
      );
      targets[name] = bone.quaternion.clone().setFromEuler(rotation);
    }
    return targets;
  }

  function capture() {
    const snapshot = {};
    for (const [name, bone] of Object.entries(bones)) {
      snapshot[name] = bone.quaternion.clone();
    }
    return snapshot;
  }

  function capturePose() {
    const pose = {};
    for (const [name, bone] of Object.entries(bones)) {
      const base = rest[name].rotation;
      pose[name] = {
        x: bone.rotation.x - base.x,
        y: bone.rotation.y - base.y,
        z: bone.rotation.z - base.z,
      };
    }
    return pose;
  }

  function interpolate(from, targets, amount) {
    for (const [name, bone] of Object.entries(bones)) {
      const start = from[name] || rest[name].quaternion;
      const target = targets[name] || rest[name].quaternion;
      // Slightly non-unit imported keys can produce dot > 1 and make slerp
      // freeze. Normalize detached, changing intermediate samples only, keeping
      // authored endpoints and identical/resting keys component-exact.
      if (start.equals(target)) {
        bone.quaternion.copy(start);
      } else if (amount > 0 && amount < 1) {
        bone.quaternion.copy(start).normalize().slerp(target.clone().normalize(), amount);
      } else {
        bone.quaternion.copy(start).slerp(target, amount);
      }
    }
    update();
  }

  function applyPose(pose) {
    const targets = createTargets(pose);
    for (const [name, bone] of Object.entries(bones)) {
      bone.quaternion.copy(targets[name]);
    }
    update();
  }

  function reset() {
    for (const [name, bone] of Object.entries(bones)) {
      bone.quaternion.copy(rest[name].quaternion);
      bone.position.copy(rest[name].position);
    }
    resetExpression();
    update();
  }

  const missing = REQUIRED_SIGNING_BONES.filter((name) => !bones[name]);
  return {
    profile,
    bones,
    rest,
    missing,
    isSigningReady: missing.length === 0,
    createTargets,
    capture,
    capturePose,
    interpolate,
    applyPose,
    reset,
    resetExpression,
    jawCalibrationAvailable,
    applyCalibratedJaw,
    manualMouthAvailable,
    manualMouthRequired,
    manualJawMaxAngle,
    applyManualMouth,
    faceControls,
    update,
  };
}

export {
  GODETTE_PROFILE,
  HUMANOID_BONES,
  LEGACY_BONE_ALIASES,
  REQUIRED_SIGNING_BONES,
  canonicalizePose,
  createHumanoidRig,
  normalizeBoneName,
  resolveProfileBones,
};
