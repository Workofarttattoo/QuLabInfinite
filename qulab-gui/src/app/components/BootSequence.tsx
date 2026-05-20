import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

interface BootSequenceProps {
  onComplete?: () => void;
}

export function BootSequence({ onComplete }: BootSequenceProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showUI, setShowUI] = useState(false);
  const sceneRef = useRef<{
    scene?: THREE.Scene;
    camera?: THREE.PerspectiveCamera;
    renderer?: THREE.WebGLRenderer;
    points?: THREE.Points;
    material?: THREE.ShaderMaterial;
    timer?: THREE.Timer;
    mouse?: THREE.Vector2;
    animationId?: number;
  }>({});

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const mouse = new THREE.Vector2(0, 0);
    sceneRef.current.mouse = mouse;

    // Initialize Three.js scene
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
    camera.position.z = 250;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const timer = new THREE.Timer();

    // Particle system
    const count = 8000;
    const positions = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    const timeOffsets = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      const r = Math.pow(Math.random(), 2.0) * 80.0;

      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);

      sizes[i] = Math.random() * 1.5 + 0.5;
      timeOffsets[i] = Math.random();
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute('aTimeOffset', new THREE.BufferAttribute(timeOffsets, 1));

    const vertexShader = `
      varying vec2 vUv;
      varying float vOpacity;
      attribute float aSize;
      attribute float aTimeOffset;
      uniform float uTime;
      uniform vec2 uMouse;

      void main() {
        vUv = uv;

        float progress = clamp(uTime / 4.0, 0.0, 1.0);
        float whirlProgress = clamp(uTime / 3.0, 0.0, 1.0);

        float angle = uTime * (3.0 + aTimeOffset * 2.0) + aTimeOffset * 25.0;
        float radius = (1.0 - whirlProgress) * (300.0 * aTimeOffset + 50.0);

        vec2 mouseOffset = uMouse * 10.0 * (1.0 - progress);

        vec3 swirlingPos = vec3(
          cos(angle) * radius + mouseOffset.x,
          sin(angle) * radius + mouseOffset.y,
          sin(angle * 0.5) * radius
        );

        vec3 finalPos = position + vec3(mouseOffset * 0.5, 0.0);

        float mixFactor = pow(progress, 3.0);
        vec3 pos = mix(swirlingPos, finalPos, mixFactor);

        vOpacity = mix(0.3, 1.0, progress);

        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);

        float sizeMod = 1.0 + (1.0 - progress) * 2.0;
        gl_PointSize = aSize * sizeMod * (350.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
      }
    `;

    const fragmentShader = `
      varying float vOpacity;
      uniform vec3 uColor;

      void main() {
        float d = distance(gl_PointCoord, vec2(0.5));
        if (d > 0.5) discard;

        float strength = 0.08 / d;
        strength = pow(strength, 1.5);

        gl_FragColor = vec4(uColor, strength * vOpacity * 0.8);
      }
    `;

    const material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uColor: { value: new THREE.Color('#00f0ff') },
        uMouse: { value: new THREE.Vector2(0, 0) }
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    sceneRef.current = { scene, camera, renderer, points, material, timer };

    // Event listeners
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    };

    window.addEventListener('resize', handleResize);
    window.addEventListener('mousemove', handleMouseMove);

    // Show UI after 3.5 seconds
    const uiTimer = setTimeout(() => setShowUI(true), 3500);

    // Animation loop
    const animate = () => {
      timer.update();
      const elapsedTime = timer.getElapsed();
      material.uniforms.uTime.value = elapsedTime;
      material.uniforms.uMouse.value.lerp(mouse, 0.05);

      points.rotation.y = elapsedTime * 0.05;
      points.rotation.z = elapsedTime * 0.02;

      renderer.render(scene, camera);
      sceneRef.current.animationId = requestAnimationFrame(animate);
    };
    animate();

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      clearTimeout(uiTimer);
      if (sceneRef.current.animationId) {
        cancelAnimationFrame(sceneRef.current.animationId);
      }
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, []);

  const handleInitialize = () => {
    if (onComplete) {
      onComplete();
    }
  };

  return (
    <div className="fixed inset-0 w-full h-full bg-[#0e0e0e] z-[9999]">
      {/* Scanlines effect */}
      <div className="scanlines fixed top-0 left-0 w-full h-full pointer-events-none z-50" style={{
        background: `linear-gradient(to bottom, rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))`,
        backgroundSize: '100% 4px, 3px 100%'
      }}></div>

      {/* WebGL Canvas Container */}
      <div ref={containerRef} className="fixed inset-0 w-full h-full"></div>

      {/* UI Overlay */}
      <div className="ui-overlay relative w-full h-full flex flex-col justify-between p-8 md:p-12 pointer-events-none z-[100]">
        {/* Top Status Bar */}
        <div className="flex justify-between items-start">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#00f0ff] rounded-full animate-pulse"></span>
              <span className="text-[10px] tracking-widest text-[#00f0ff] font-bold font-['JetBrains_Mono']">CORE_STABLE_V1.0</span>
            </div>
            <div className="text-[9px] text-[#849495] uppercase tracking-widest opacity-60 font-['JetBrains_Mono']">System Ready // Latency 0.4ms</div>
          </div>
          <div className="text-right">
            <div className="text-[10px] text-[#00f0ff]/60 tracking-widest font-['JetBrains_Mono']">LOC: 34.0522° N, 118.2437° W</div>
            <div className="text-[10px] text-[#849495]/40 font-['JetBrains_Mono']">AUTH_LEVEL: OVERRIDE</div>
          </div>
        </div>

        {/* Center Branding */}
        <div className="flex-1 flex flex-col items-center justify-center text-center">
          <div className={`space-y-4 transition-opacity duration-1000 ${showUI ? 'opacity-100' : 'opacity-0'}`}>
            <h1 className="font-['Space_Grotesk'] text-5xl md:text-7xl lg:text-8xl font-bold tracking-tighter text-[#00f0ff] uppercase" style={{
              textShadow: '0 0 10px rgba(0, 240, 255, 0.5), 0 0 20px rgba(0, 240, 255, 0.3)'
            }}>
              QuLab Infinite
            </h1>
            <p className="font-['Space_Grotesk'] text-lg md:text-xl text-[#849495]/80 tracking-[0.4em] uppercase font-light">
              Design the Future
            </p>
            <div className="pt-12 pointer-events-auto">
              <button
                onClick={handleInitialize}
                className="px-10 py-4 text-xs font-bold tracking-[0.3em] uppercase text-[#00f0ff] bg-transparent rounded-sm border border-[#00f0ff]/30 hover:bg-[#00f0ff]/10 hover:border-[#00f0ff] hover:scale-105 transition-all duration-300 font-['JetBrains_Mono']"
                style={{
                  boxShadow: '0 0 0 rgba(0, 240, 255, 0.4)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.boxShadow = '0 0 20px rgba(0, 240, 255, 0.4)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = '0 0 0 rgba(0, 240, 255, 0.4)';
                }}
              >
                Initialize System
              </button>
            </div>
          </div>
        </div>

        {/* Bottom Metadata */}
        <div className="flex justify-between items-end">
          <div className="max-w-xs space-y-2">
            <div className="h-[1px] w-12 bg-[#00f0ff]/30"></div>
            <p className="text-[10px] text-[#849495] leading-relaxed opacity-50 font-['JetBrains_Mono']">
              Proprietary neural-link interface for architectural synthesis and advanced geometric derivation. Unauthorized access is strictly prohibited.
            </p>
          </div>
          <div className="text-right font-['JetBrains_Mono'] text-[10px] text-[#849495]/60">
            © 2026 QULAB_INDUSTRIES // NEURAL_NET_V4
          </div>
        </div>
      </div>
    </div>
  );
}
