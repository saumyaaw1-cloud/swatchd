import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const WORLD_BASE_Y = -0.55;

function createDisc(radius, depth, material) {
  const disc = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, depth, 96), material);
  disc.rotation.x = Math.PI / 2;
  disc.castShadow = true;
  disc.receiveShadow = true;
  return disc;
}

export function initCompactScene(container) {
  if (!container) return null;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0, 0.05, 7);

  const renderer = new THREE.WebGLRenderer({
    alpha: true,
    antialias: true,
    powerPreference: "high-performance"
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const environmentGenerator = new THREE.PMREMGenerator(renderer);
  scene.environment = environmentGenerator.fromScene(new RoomEnvironment(), 0.05).texture;
  environmentGenerator.dispose();
  renderer.domElement.setAttribute("aria-hidden", "true");
  container.prepend(renderer.domElement);

  const world = new THREE.Group();
  const compact = new THREE.Group();
  world.add(compact);
  scene.add(world);

  scene.add(new THREE.HemisphereLight(0xfff9f1, 0x6d234c, 2.5));

  const keyLight = new THREE.DirectionalLight(0xffffff, 4.2);
  keyLight.position.set(-3.5, 4.5, 5);
  scene.add(keyLight);

  const pinkLight = new THREE.PointLight(0xff7db5, 20, 11, 2);
  pinkLight.position.set(3, 0.8, 4);
  scene.add(pinkLight);

  const shellMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xffb6d2,
    roughness: 0.34,
    metalness: 0.04,
    clearcoat: 1,
    clearcoatRoughness: 0.2
  });
  const rimMaterial = new THREE.MeshStandardMaterial({
    color: 0xfff8e8,
    roughness: 0.28,
    metalness: 0.08
  });
  const goldMaterial = new THREE.MeshStandardMaterial({
    color: 0xffdc62,
    roughness: 0.3,
    metalness: 0.34
  });
  const mirrorBackingMaterial = new THREE.MeshStandardMaterial({
    color: 0xff3b91,
    emissive: 0xff146f,
    emissiveIntensity: 1.05,
    roughness: 0.28,
    metalness: 0.08
  });
  const mirrorMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xffb7d2,
    roughness: 0.075,
    metalness: 0.08,
    transmission: 0.42,
    thickness: 0.22,
    ior: 1.48,
    transparent: true,
    opacity: 0.94,
    clearcoat: 1,
    clearcoatRoughness: 0.025,
    envMapIntensity: 1.2,
    specularIntensity: 1,
    specularColor: 0xffffff
  });

  const base = new THREE.Group();
  compact.add(base);

  const baseShell = createDisc(1.28, 0.2, shellMaterial);
  baseShell.position.z = -0.04;
  base.add(baseShell);

  const baseRim = new THREE.Mesh(new THREE.TorusGeometry(1.18, 0.075, 20, 96), rimMaterial);
  baseRim.position.z = 0.09;
  base.add(baseRim);

  const mirrorBacking = createDisc(1.07, 0.035, mirrorBackingMaterial);
  mirrorBacking.position.z = 0.095;
  base.add(mirrorBacking);

  const mirror = createDisc(1.08, 0.035, mirrorMaterial);
  mirror.position.z = 0.13;
  base.add(mirror);

  const baseGlow = new THREE.PointLight(0xff3c96, 0, 3.8, 2);
  baseGlow.position.set(0, 0, 0.52);
  base.add(baseGlow);

  const latch = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.16, 0.18), goldMaterial);
  latch.position.set(0, -1.27, 0.02);
  latch.rotation.x = 0.08;
  base.add(latch);

  const hingeBar = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.62, 28), goldMaterial);
  hingeBar.rotation.z = Math.PI / 2;
  hingeBar.position.set(0, 1.19, 0.04);
  compact.add(hingeBar);

  const lidHinge = new THREE.Group();
  lidHinge.position.set(0, 1.16, 0.08);
  compact.add(lidHinge);

  const lid = new THREE.Group();
  lid.position.set(0, -1.16, 0.14);
  lidHinge.add(lid);

  const lidShell = createDisc(1.29, 0.2, shellMaterial);
  lid.add(lidShell);

  const lidMirrorBacking = createDisc(1.07, 0.035, mirrorBackingMaterial);
  lidMirrorBacking.position.z = -0.12;
  lid.add(lidMirrorBacking);

  const lidMirror = createDisc(1.08, 0.035, mirrorMaterial);
  lidMirror.position.z = -0.155;
  lid.add(lidMirror);

  const lidGlow = new THREE.PointLight(0xff3c96, 0, 3.2, 2);
  lidGlow.position.set(0, 0, -0.42);
  lid.add(lidGlow);

  const lidInnerRim = new THREE.Mesh(new THREE.TorusGeometry(1.18, 0.075, 20, 96), rimMaterial);
  lidInnerRim.position.z = -0.17;
  lid.add(lidInnerRim);

  const lidRim = new THREE.Mesh(new THREE.TorusGeometry(1.19, 0.075, 20, 96), rimMaterial);
  lidRim.position.z = 0.12;
  lid.add(lidRim);

  compact.rotation.x = -0.07;
  compact.rotation.z = -0.035;
  compact.scale.setScalar(0.78);
  container.classList.add("is-ready");

  let targetOpen = 0;
  let openAmount = 0;
  let pointerX = 0;
  let pointerY = 0;
  let hasInteracted = false;

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const previewOpenTimer = prefersReducedMotion
      ? null
    : window.setTimeout(() => {
        if (!hasInteracted) targetOpen = 1;
      }, 650);
  const previewCloseTimer = prefersReducedMotion
    ? null
    : window.setTimeout(() => {
        if (!hasInteracted) targetOpen = 0;
      }, 2550);

  const resize = () => {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();

  const clock = new THREE.Clock();
  const render = () => {
    const delta = Math.min(clock.getDelta(), 0.05);
    const response = 1 - Math.exp(-delta * 5.2);
    openAmount = THREE.MathUtils.lerp(openAmount, targetOpen, response);

    const easedOpen = openAmount * openAmount * (3 - 2 * openAmount);
    lidHinge.rotation.x = -easedOpen * Math.PI * 0.62;
    baseGlow.intensity = easedOpen * 4.2;
    lidGlow.intensity = easedOpen * 3.2;
    world.rotation.y = THREE.MathUtils.lerp(world.rotation.y, pointerX * 0.16, 0.055);
    world.rotation.x = THREE.MathUtils.lerp(world.rotation.x, -pointerY * 0.09, 0.055);
    world.position.y = WORLD_BASE_Y + Math.sin(clock.elapsedTime * 1.15) * 0.025;
    renderer.render(scene, camera);
    requestAnimationFrame(render);
  };
  render();

  console.info("[swatchd-compact] Three.js compact mirror ready");

  return {
    setOpen(isOpen) {
      hasInteracted = true;
      if (previewOpenTimer) window.clearTimeout(previewOpenTimer);
      if (previewCloseTimer) window.clearTimeout(previewCloseTimer);
      targetOpen = isOpen ? 1 : 0;
    },
    setPointer(x, y) {
      pointerX = THREE.MathUtils.clamp(x, -0.5, 0.5);
      pointerY = THREE.MathUtils.clamp(y, -0.5, 0.5);
    },
    get ready() {
      return true;
    }
  };
}
