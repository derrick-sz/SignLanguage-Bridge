import * as THREE from "/animation/vendor/three.module.js";
import { GLTFLoader } from "/animation/vendor/jsm/loaders/GLTFLoader.js";
import { createHumanoidRig } from "./rig.js";
import { createT08Retarget } from "./retarget.js";
import { stabilizeQuaternionTrack } from "./stabilization.js";
import { PlaybackClock, signingCamera } from "./player-core.mjs";

export class MotionStage {
  constructor(container, onTick) {
    this.container = container; this.onTick = onTick;
    this.clock = new PlaybackClock(); this.profile = "improved"; this.view = "front";
    this.cache = new Map(); this.disposed = false;
    this.ready = this.init();
  }
  async init() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);
    this.renderer.domElement.setAttribute("aria-label", "手语人物三维画面");
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.05, 30);
    this.hemi = new THREE.HemisphereLight(0xffffff, 0x8094ac, 1.6);
    this.key = new THREE.DirectionalLight(0xfff5e8, 2.5);
    this.key.position.set(2.5, 4, 4); this.key.castShadow = true;
    this.key.shadow.mapSize.set(1024, 1024);
    this.key.shadow.camera.left = -2; this.key.shadow.camera.right = 2;
    this.key.shadow.camera.top = 3; this.key.shadow.camera.bottom = -1;
    this.fill = new THREE.DirectionalLight(0xe1eeff, 1.0);
    this.fill.position.set(-3, 2, 2);
    this.rim = new THREE.DirectionalLight(0xc4dcff, 1.7);
    this.rim.position.set(0, 3, -3);
    this.scene.add(this.hemi, this.key, this.fill, this.rim);
    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(20, 20),
      new THREE.ShadowMaterial({ opacity: 0.18 }));
    this.floor.rotation.x = -Math.PI / 2; this.floor.receiveShadow = true;
    this.scene.add(this.floor);
    const gltf = await new GLTFLoader().loadAsync("/animation/vendor/model.glb");
    this.model = gltf.scene;
    if (this.disposed) { this.disposeModel(); return; }
    this.model.traverse(node => {
      if (!node.isMesh) return;
      node.castShadow = true; node.receiveShadow = true;
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      materials.forEach(material => {
        material.userData.quietPalette = { value: 1 };
        material.onBeforeCompile = shader => {
          shader.uniforms.quietPalette = material.userData.quietPalette;
          shader.fragmentShader = "uniform float quietPalette;\n" + shader.fragmentShader.replace(
            "#include <color_fragment>",
            `#include <color_fragment>
            float blueArea = smoothstep(0.02, 0.16, diffuseColor.b - diffuseColor.r);
            float luminance = dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722));
            vec3 quietBlue = vec3(0.07, 0.115, 0.17) * (0.7 + luminance);
            diffuseColor.rgb = mix(diffuseColor.rgb, quietBlue, blueArea * quietPalette * 0.84);`);
        };
        material.customProgramCacheKey = () => "slbridge-quiet-palette-v1";
      });
    });
    this.scene.add(this.model);
    this.rig = createHumanoidRig(this.model);
    if (!this.rig.isSigningReady) throw new Error("角色骨架缺少必要手语关节");
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.setView("front"); this.setProfile("improved"); this.resize();
    const render = now => {
      if (this.disposed) return;
      this.refresh(now);
      this.raf = requestAnimationFrame(render);
    };
    this.raf = requestAnimationFrame(render);
  }
  resize() {
    const { clientWidth: width, clientHeight: height } = this.container;
    if (!width || !height) return;
    this.renderer.setSize(width, height); this.camera.aspect = width / height;
    this.setView(this.view);
  }
  setView(view) {
    this.view = view;
    if (!this.camera) return;
    const frame = this.bounds ? signingCamera(this.bounds, this.camera.aspect) : { centerY: 1.25, distance: 2.05 };
    const center = new THREE.Vector3(0, view === "hands" ? 1.3 : frame.centerY, 0);
    const distance = view === "hands" ? 1.75 : frame.distance;
    const offset = view === "side" ? new THREE.Vector3(1.7, 0.04, 1.55) : new THREE.Vector3(0, 0.01, distance);
    this.camera.position.copy(center).add(offset);
    this.camera.lookAt(center);
    this.camera.updateProjectionMatrix();
    this.refresh();
  }
  setProfile(profile) {
    this.profile = profile;
    if (!this.scene) return;
    const improved = profile === "improved";
    this.scene.background = new THREE.Color(improved ? 0x152230 : 0xe4ebee);
    this.renderer.toneMapping = improved ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
    this.hemi.intensity = improved ? 1.5 : 2.0;
    this.key.intensity = improved ? 2.5 : 2.6;
    this.fill.intensity = improved ? 0.9 : 1.2;
    this.rim.intensity = improved ? 1.6 : 0;
    this.floor.visible = improved;
    this.model?.traverse(node => {
      const materials = node.isMesh ? (Array.isArray(node.material) ? node.material : [node.material]) : [];
      materials.forEach(m => { m.userData.quietPalette.value = improved ? 1 : 0; });
    });
    if (this.tracks) this.track = improved ? this.tracks.improved : this.tracks.original;
    this.refresh();
  }
  async loadMotion(data, id, isCurrent = () => true) {
    await this.ready;
    if (this.disposed || !isCurrent()) return false;
    let tracks = this.cache.get(id);
    if (!tracks) {
      const retarget = createT08Retarget(this.rig, this.model, data);
      const original = retarget.compile(data.clip, { stabilize: false });
      const improved = stabilizeQuaternionTrack(original, data.clip.fps, "quaternion-temporal-adaptive-v1");
      let corrected = 0;
      original.forEach((frame, i) => {
        if (Object.keys(frame).some(k => frame[k].angleTo(improved[i][k]) > 1e-6)) corrected++;
      });
      tracks = { original, improved, corrected, mappedBones: retarget.mappedBones };
      const snapshot = this.rig.capture();
      const bounds = { minX: -0.28, maxX: 0.28, minY: 0.72, maxY: 1.86 };
      // Fit once to the target rig's complete motion envelope; never chase hands per frame.
      for (let i = 0; i < original.length; i++) {
        this.rig.interpolate(original[i], original[i], 1);
        this.model.updateMatrixWorld(true);
        for (const name of ["leftHand", "rightHand", "leftIndexDistal", "rightIndexDistal", "head"]) {
          const p = this.rig.bones[name]?.getWorldPosition(new THREE.Vector3());
          if (!p) continue;
          bounds.minX = Math.min(bounds.minX, p.x); bounds.maxX = Math.max(bounds.maxX, p.x);
          bounds.minY = Math.min(bounds.minY, p.y - 0.05); bounds.maxY = Math.max(bounds.maxY, p.y + 0.08);
        }
      }
      this.rig.interpolate(snapshot, snapshot, 1);
      tracks.bounds = bounds;
      this.cache.set(id, tracks);
      if (this.cache.size > 6) this.cache.delete(this.cache.keys().next().value);
    }
    if (!isCurrent()) return false;
    this.tracks = tracks; this.track = tracks[this.profile];
    this.fps = data.clip.fps; this.clip = data.clip;
    this.bounds = tracks.bounds; this.setView(this.view);
    this.clock.load(data.clip.frames / data.clip.fps);
    this.refresh();
    return true;
  }
  applyTime(time) {
    if (!this.track?.length) return;
    const position = Math.min(time * this.fps, this.track.length - 1);
    const from = Math.floor(position), to = Math.min(from + 1, this.track.length - 1);
    this.rig.interpolate(this.track[from], this.track[to], position - from);
  }
  refresh(now = performance.now()) {
    if (this.disposed || !this.renderer) return;
    const time = this.clock.time(now);
    this.applyTime(time);
    this.renderer.render(this.scene, this.camera);
    this.onTick?.({ time, duration: this.clock.duration, playing: this.clock.playing });
  }
  play() { this.clock.play(performance.now()); this.refresh(); }
  pause() { this.clock.pause(performance.now()); this.refresh(); }
  seek(time) { this.clock.seek(time, performance.now()); this.refresh(); }
  setSpeed(speed) { this.clock.setSpeed(speed, performance.now()); this.refresh(); }
  stop() {
    this.clock.stop(); this.clock.duration = 0;
    this.track = null; this.tracks = null; this.clip = null;
    this.rig?.reset(); this.refresh();
  }
  disposeModel() {
    this.model?.traverse(node => {
      if (!node.isMesh) return;
      node.geometry.dispose();
      (Array.isArray(node.material) ? node.material : [node.material]).forEach(m => {
        for (const value of Object.values(m)) if (value?.isTexture) value.dispose();
        m.dispose();
      });
    });
  }
  dispose() {
    this.disposed = true; cancelAnimationFrame(this.raf);
    this.resizeObserver?.disconnect(); this.disposeModel(); this.floor?.geometry.dispose();
    this.floor?.material.dispose(); this.renderer?.dispose();
  }
}
