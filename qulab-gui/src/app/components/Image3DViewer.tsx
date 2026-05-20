import { useState, useRef, useEffect } from 'react';

interface Image3DViewerProps {
  imageUrl?: string;
  src?: string;
  alt?: string;
  className?: string;
  autoRotate?: boolean;
}

export function Image3DViewer({
  imageUrl,
  src,
  alt = '3D Model',
  className = '',
  autoRotate = false,
}: Image3DViewerProps) {
  const resolvedUrl = imageUrl ?? src ?? '';
  const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 });
  const [perspective, setPerspective] = useState(1000);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-rotation effect
  useEffect(() => {
    if (!autoRotate) return;

    const interval = setInterval(() => {
      setRotation(prev => ({
        ...prev,
        y: (prev.y + 1) % 360
      }));
    }, 50);

    return () => clearInterval(interval);
  }, [autoRotate]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;

    const deltaX = e.clientX - dragStart.x;
    const deltaY = e.clientY - dragStart.y;

    setRotation(prev => ({
      x: prev.x + deltaY * 0.5,
      y: prev.y + deltaX * 0.5,
      z: prev.z
    }));

    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setPerspective(prev => Math.max(500, Math.min(2000, prev + e.deltaY)));
  };

  const resetView = () => {
    setRotation({ x: 0, y: 0, z: 0 });
    setPerspective(1000);
  };

  return (
    <div
      ref={containerRef}
      className={`relative ${className}`}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
      style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
    >
      {/* 3D Transform Container */}
      <div
        className="w-full h-full"
        style={{
          perspective: `${perspective}px`,
          transformStyle: 'preserve-3d',
        }}
      >
        <div
          className="w-full h-full transition-transform duration-100"
          style={{
            transform: `rotateX(${rotation.x}deg) rotateY(${rotation.y}deg) rotateZ(${rotation.z}deg)`,
            transformStyle: 'preserve-3d',
          }}
        >
          {/* Front face */}
          <img
            src={resolvedUrl}
            alt={alt}
            className="w-full h-full object-contain opacity-90"
            style={{
              transform: 'translateZ(0px)',
              backfaceVisibility: 'hidden',
            }}
            draggable={false}
          />

          {/* Create depth layers for pseudo-3D effect */}
          {[...Array(5)].map((_, i) => (
            <img
              key={i}
              src={resolvedUrl}
              alt=""
              className="absolute inset-0 w-full h-full object-contain opacity-20"
              style={{
                transform: `translateZ(-${(i + 1) * 20}px)`,
                filter: `blur(${i * 2}px)`,
                backfaceVisibility: 'hidden',
              }}
              draggable={false}
            />
          ))}
        </div>
      </div>

      {/* 3D Control Overlay */}
      <div className="absolute top-2 right-2 flex flex-col gap-2 z-20">
        <button
          onClick={() => setRotation(prev => ({ ...prev, x: prev.x + 45 }))}
          className="w-8 h-8 rounded-full glass-panel flex items-center justify-center text-white hover:bg-[#137fec]/20 transition-colors"
          title="Rotate X+"
        >
          <span className="material-symbols-outlined text-sm">rotate_right</span>
        </button>
        <button
          onClick={() => setRotation(prev => ({ ...prev, y: prev.y + 45 }))}
          className="w-8 h-8 rounded-full glass-panel flex items-center justify-center text-white hover:bg-[#137fec]/20 transition-colors"
          title="Rotate Y+"
        >
          <span className="material-symbols-outlined text-sm">3d_rotation</span>
        </button>
        <button
          onClick={() => setRotation(prev => ({ ...prev, z: prev.z + 45 }))}
          className="w-8 h-8 rounded-full glass-panel flex items-center justify-center text-white hover:bg-[#137fec]/20 transition-colors"
          title="Rotate Z+"
        >
          <span className="material-symbols-outlined text-sm">autorenew</span>
        </button>
        <button
          onClick={resetView}
          className="w-8 h-8 rounded-full glass-panel flex items-center justify-center text-white hover:bg-[#137fec]/20 transition-colors"
          title="Reset View"
        >
          <span className="material-symbols-outlined text-sm">restart_alt</span>
        </button>
      </div>

      {/* Interaction Hint */}
      {!isDragging && rotation.x === 0 && rotation.y === 0 && (
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] text-[#b9cacb] bg-black/50 px-3 py-1 rounded-full pointer-events-none">
          Drag to rotate • Scroll to zoom
        </div>
      )}

      {/* Perspective Info */}
      <div className="absolute bottom-2 right-2 text-[8px] font-mono text-[#b9cacb] bg-black/50 px-2 py-1 rounded">
        X:{rotation.x.toFixed(0)}° Y:{rotation.y.toFixed(0)}° Z:{rotation.z.toFixed(0)}°
      </div>
    </div>
  );
}
