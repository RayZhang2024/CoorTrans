import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import "./style.css";

type PlanePickerProps = {
  width?: number;
  args: {
    p0: number[];
    u: number[];
    v: number[];
    umin: number;
    umax: number;
    vmin: number;
    vmax: number;
    width?: number;
    height?: number;
    segments?: number[][][];
    picks?: number[][];
    picks_rev?: number;
    live_preview?: boolean;
  };
};

const PlanePicker = ({ args, width: componentWidth }: PlanePickerProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{
    active: boolean;
    pointerId: number | null;
    startClientX: number;
    startClientY: number;
    startPanX: number;
    startPanY: number;
    moved: boolean;
  }>({
    active: false,
    pointerId: null,
    startClientX: 0,
    startClientY: 0,
    startPanX: 0,
    startPanY: 0,
    moved: false,
  });
  const [cursor, setCursor] = useState<{ u: number; v: number } | null>(null);
  const eventSeqRef = useRef<number>(0);
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [localPicks, setLocalPicks] = useState<number[][]>([]);
  const [pendingPicks, setPendingPicks] = useState<number[][]>([]);

  const requestedWidth = args?.width ?? 420;
  const availableWidth =
    ((typeof componentWidth === "number" ? componentWidth : window.innerWidth) ||
      requestedWidth) - 20;
  const width = Math.max(320, Math.min(requestedWidth, availableWidth));
  const height = args?.height ?? 420;

  useEffect(() => {
    setLocalPicks(args?.picks ?? []);
    setPendingPicks([]);
  }, [args?.picks_rev]);

  const viewTransform = () => {
    const umin = args?.umin ?? -1;
    const umax = args?.umax ?? 1;
    const vmin = args?.vmin ?? -1;
    const vmax = args?.vmax ?? 1;
    const du = Math.max(umax - umin, 1e-9);
    const dv = Math.max(vmax - vmin, 1e-9);
    const pad = 16;
    const baseScale = Math.min((width - 2 * pad) / du, (height - 2 * pad) / dv);
    const scale = baseScale * zoom;
    const drawW = du * scale;
    const drawH = dv * scale;
    const ox = (width - drawW) / 2 + pan.x;
    const oy = (height - drawH) / 2 + pan.y;
    return { umin, umax, vmin, vmax, du, dv, scale, drawW, drawH, ox, oy };
  };

  useEffect(() => {
    Streamlit.setComponentReady();
    Streamlit.setFrameHeight(height + 96);
  }, [height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const stopWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
    };
    canvas.addEventListener("wheel", stopWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", stopWheel);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Outer canvas frame
    ctx.strokeStyle = "#d0d0d0";
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, width, height);

    const segments: number[][][] = args?.segments ?? [];
    const picks: number[][] = localPicks;
    const { umin, umax, vmin, vmax, scale, drawW, drawH, ox, oy } = viewTransform();
    const toX = (u: number) => ox + (u - umin) * scale;
    const toY = (v: number) => oy + (vmax - v) * scale;

    const drawArrow2D = (
      x0: number,
      y0: number,
      x1: number,
      y1: number,
      color: string,
      label: string
    ) => {
      const dx = x1 - x0;
      const dy = y1 - y0;
      const len = Math.hypot(dx, dy);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2.5;
      if (len < 1e-6) {
        ctx.beginPath();
        ctx.arc(x0, y0, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(label, x0 + 6, y0 - 4);
        return;
      }
      const ux = dx / len;
      const uy = dy / len;
      const head = Math.min(10, Math.max(6, len * 0.28));
      const leftX = x1 - ux * head - uy * (head * 0.55);
      const leftY = y1 - uy * head + ux * (head * 0.55);
      const rightX = x1 - ux * head + uy * (head * 0.55);
      const rightY = y1 - uy * head - ux * (head * 0.55);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(leftX, leftY);
      ctx.lineTo(rightX, rightY);
      ctx.closePath();
      ctx.fill();
      ctx.fillText(label, x1 + 4, y1 - 4);
    };

    // Draw active plane bounds and a light UV grid (with clearer zero lines).
    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, drawW, drawH);

    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, drawW, drawH);
    ctx.clip();

    const gridDivs = 6;
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    for (let i = 1; i < gridDivs; i += 1) {
      const uTick = umin + ((umax - umin) * i) / gridDivs;
      const xTick = toX(uTick);
      ctx.beginPath();
      ctx.moveTo(xTick, oy);
      ctx.lineTo(xTick, oy + drawH);
      ctx.stroke();
    }
    for (let i = 1; i < gridDivs; i += 1) {
      const vTick = vmin + ((vmax - vmin) * i) / gridDivs;
      const yTick = toY(vTick);
      ctx.beginPath();
      ctx.moveTo(ox, yTick);
      ctx.lineTo(ox + drawW, yTick);
      ctx.stroke();
    }

    ctx.strokeStyle = "#6b7280";
    ctx.lineWidth = 1.5;
    if (umin <= 0 && 0 <= umax) {
      const xZero = toX(0);
      ctx.beginPath();
      ctx.moveTo(xZero, oy);
      ctx.lineTo(xZero, oy + drawH);
      ctx.stroke();
    }
    if (vmin <= 0 && 0 <= vmax) {
      const yZero = toY(0);
      ctx.beginPath();
      ctx.moveTo(ox, yZero);
      ctx.lineTo(ox + drawW, yZero);
      ctx.stroke();
    }
    ctx.restore();

    // Inset XYZ triad projected into the current plane basis.
    const insetW = 96;
    const insetH = 96;
    const insetX = width - insetW - 10;
    const insetY = 10;
    ctx.fillStyle = "rgba(255,255,255,0.86)";
    ctx.strokeStyle = "rgba(0,0,0,0.15)";
    ctx.lineWidth = 1;
    ctx.fillRect(insetX, insetY, insetW, insetH);
    ctx.strokeRect(insetX, insetY, insetW, insetH);
    const cX = insetX + 30;
    const cY = insetY + 66;
    const triadScale = 28;
    const uDir = args?.u ?? [1, 0, 0];
    const vDir = args?.v ?? [0, 1, 0];
    const xyzAxes = [
      { label: "X", vec: [1, 0, 0], color: "#d62728" },
      { label: "Y", vec: [0, 1, 0], color: "#2ca02c" },
      { label: "Z", vec: [0, 0, 1], color: "#1f77b4" },
    ];
    ctx.strokeStyle = "#9ca3af";
    ctx.beginPath();
    ctx.arc(cX, cY, 2.5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.font = "12px Arial, sans-serif";
    xyzAxes.forEach(({ label, vec, color }) => {
      const uProj = vec[0] * uDir[0] + vec[1] * uDir[1] + vec[2] * uDir[2];
      const vProj = vec[0] * vDir[0] + vec[1] * vDir[1] + vec[2] * vDir[2];
      const mag = Math.hypot(uProj, vProj);
      const nx = mag > 1e-6 ? uProj / mag : 0;
      const ny = mag > 1e-6 ? -vProj / mag : 0;
      const len = mag > 1e-6 ? triadScale * Math.min(1, mag) : 0;
      drawArrow2D(cX, cY, cX + nx * len, cY + ny * len, color, label);
    });

    ctx.strokeStyle = "#ff7f0e";
    ctx.lineWidth = 2;
    segments.forEach((seg) => {
      if (seg.length < 2) return;
      const [a, b] = seg;
      ctx.beginPath();
      ctx.moveTo(toX(a[0]), toY(a[1]));
      ctx.lineTo(toX(b[0]), toY(b[1]));
      ctx.stroke();
    });

    ctx.strokeStyle = "#2ca02c";
    ctx.lineWidth = 2;
    const crossSize = 6;
    picks.forEach((pt) => {
      if (pt.length < 2) return;
      const px = toX(pt[0]);
      const py = toY(pt[1]);
      ctx.beginPath();
      ctx.moveTo(px - crossSize, py);
      ctx.lineTo(px + crossSize, py);
      ctx.moveTo(px, py - crossSize);
      ctx.lineTo(px, py + crossSize);
      ctx.stroke();
    });
  }, [
    width,
    height,
    zoom,
    args?.umin,
    args?.umax,
    args?.vmin,
    args?.vmax,
    args?.u,
    args?.v,
    args?.segments,
    localPicks,
    pan,
  ]);

  const coordsText = useMemo(() => {
    if (!cursor) return "Move cursor on plane to see live coordinates.";
    const p0 = args?.p0 ?? [0, 0, 0];
    const uDir = args?.u ?? [1, 0, 0];
    const vDir = args?.v ?? [0, 1, 0];
    const x = p0[0] + cursor.u * uDir[0] + cursor.v * vDir[0];
    const y = p0[1] + cursor.u * uDir[1] + cursor.v * vDir[1];
    const z = p0[2] + cursor.u * uDir[2] + cursor.v * vDir[2];
    return `u=${cursor.u.toFixed(3)}, v=${cursor.v.toFixed(3)} | x=${x.toFixed(
      3
    )}, y=${y.toFixed(3)}, z=${z.toFixed(3)} mm`;
  }, [args?.p0, args?.u, args?.v, cursor]);

  const cursorUV = (
    evt: React.MouseEvent<HTMLCanvasElement> | React.PointerEvent<HTMLCanvasElement>
  ) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;
    const { umin, vmax, scale, drawW, drawH, ox, oy } = viewTransform();
    const xClamped = Math.min(Math.max(x, ox), ox + drawW);
    const yClamped = Math.min(Math.max(y, oy), oy + drawH);
    const u = umin + (xClamped - ox) / scale;
    const v = vmax - (yClamped - oy) / scale;
    return { u, v };
  };

  const addPick = (u: number, v: number) => {
    setLocalPicks((prev) => [...prev, [u, v]]);
    setPendingPicks((prev) => {
      const next = [...prev, [u, v]];
      if (args?.live_preview) {
        eventSeqRef.current += 1;
        Streamlit.setComponentValue({
          preview_picks: next,
          preview_id: eventSeqRef.current,
        });
      }
      return next;
    });
    setCursor({ u, v });
  };

  const handlePointerDown = (evt: React.PointerEvent<HTMLCanvasElement>) => {
    if (evt.button !== 0) return;
    evt.preventDefault();
    evt.stopPropagation();
    evt.currentTarget.setPointerCapture(evt.pointerId);
    dragRef.current = {
      active: true,
      pointerId: evt.pointerId,
      startClientX: evt.clientX,
      startClientY: evt.clientY,
      startPanX: pan.x,
      startPanY: pan.y,
      moved: false,
    };
    setIsDragging(false);
    setCursor(cursorUV(evt));
  };

  const handlePointerMove = (evt: React.PointerEvent<HTMLCanvasElement>) => {
    setCursor(cursorUV(evt));
    const drag = dragRef.current;
    if (!drag.active || drag.pointerId !== evt.pointerId) return;
    evt.preventDefault();
    evt.stopPropagation();
    const dx = evt.clientX - drag.startClientX;
    const dy = evt.clientY - drag.startClientY;
    const moved = Math.hypot(dx, dy) > 3;
    if (moved && !drag.moved) {
      drag.moved = true;
      setIsDragging(true);
    }
    setPan({ x: drag.startPanX + dx, y: drag.startPanY + dy });
  };

  const handlePointerUp = (evt: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag.active || drag.pointerId !== evt.pointerId) return;
    evt.preventDefault();
    evt.stopPropagation();
    if (evt.currentTarget.hasPointerCapture(evt.pointerId)) {
      evt.currentTarget.releasePointerCapture(evt.pointerId);
    }
    const dx = evt.clientX - drag.startClientX;
    const dy = evt.clientY - drag.startClientY;
    const moved = drag.moved || Math.hypot(dx, dy) > 3;
    dragRef.current = {
      ...dragRef.current,
      active: false,
      pointerId: null,
      moved: false,
    };
    setIsDragging(false);
    const { u, v } = cursorUV(evt);
    setCursor({ u, v });
    if (!moved) {
      addPick(u, v);
    }
  };

  const handlePointerCancel = (evt: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (drag.active && drag.pointerId === evt.pointerId) {
      dragRef.current = {
        ...dragRef.current,
        active: false,
        pointerId: null,
        moved: false,
      };
      setIsDragging(false);
    }
  };

  const handleWheel = (evt: React.WheelEvent<HTMLCanvasElement>) => {
    evt.preventDefault();
    evt.stopPropagation();
    const delta = evt.deltaY < 0 ? 1.1 : 0.9;
    setZoom((z) => Math.min(8, Math.max(0.3, z * delta)));
  };

  return (
    <div
      className="container"
      onWheelCapture={(evt) => {
        evt.preventDefault();
        evt.stopPropagation();
      }}
    >
      <div className="caption">
        <button type="button" onClick={() => setZoom((z) => Math.min(8, z * 1.2))}>
          Zoom In
        </button>{" "}
        <button type="button" onClick={() => setZoom((z) => Math.max(0.3, z / 1.2))}>
          Zoom Out
        </button>{" "}
        <button
          type="button"
          onClick={() => {
            setZoom(1);
            setPan({ x: 0, y: 0 });
          }}
        >
          Reset
        </button>{" "}
        <button
          type="button"
          onClick={() => {
            if (pendingPicks.length === 0) return;
            Streamlit.setComponentValue({
              picks: pendingPicks,
              commit_id: Date.now(),
            });
          }}
        >
          Commit Picks ({pendingPicks.length})
        </button>{" "}
        <button
          type="button"
          onClick={() => {
            setLocalPicks((prev) => prev.slice(0, -1));
            setPendingPicks((prev) => {
              const next = prev.slice(0, -1);
              if (args?.live_preview) {
                eventSeqRef.current += 1;
                Streamlit.setComponentValue({
                  preview_picks: next,
                  preview_id: eventSeqRef.current,
                });
              }
              return next;
            });
          }}
          disabled={localPicks.length === 0}
        >
          Undo Last
        </button>{" "}
        zoom={zoom.toFixed(2)}x
      </div>
      <canvas
        ref={canvasRef}
        className={`canvas${isDragging ? " dragging" : ""}`}
        width={width}
        height={height}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerCancel}
        onWheel={handleWheel}
      />
      <div className="caption">
        Click to place markers. Drag to pan. Then press Commit Picks.
      </div>
      <div className="caption">{coordsText}</div>
    </div>
  );
};

const WrappedPlanePicker = withStreamlitConnection(PlanePicker);
export default WrappedPlanePicker;

const root = ReactDOM.createRoot(document.getElementById("root")!);
root.render(<WrappedPlanePicker />);
