import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import "./style.css";

type PlanePickerProps = {
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
  };
};

const PlanePicker = ({ args }: PlanePickerProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [cursor, setCursor] = useState<{ u: number; v: number } | null>(null);
  const [zoom, setZoom] = useState<number>(1);
  const [localPicks, setLocalPicks] = useState<number[][]>([]);
  const [pendingPicks, setPendingPicks] = useState<number[][]>([]);

  const width = args?.width ?? 420;
  const height = args?.height ?? 420;

  useEffect(() => {
    setLocalPicks(args?.picks ?? []);
    setPendingPicks([]);
  }, [args?.picks_rev]);

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
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    // Draw border and axes
    ctx.strokeStyle = "#d0d0d0";
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, width, height);

    ctx.strokeStyle = "#e0e0e0";
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.stroke();

    const segments: number[][][] = args?.segments ?? [];
    const picks: number[][] = localPicks;
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
    const ox = (width - drawW) / 2;
    const oy = (height - drawH) / 2;
    const toX = (u: number) => ox + (u - umin) * scale;
    const toY = (v: number) => oy + (vmax - v) * scale;

    // Draw active plane bounds (true aspect ratio)
    ctx.strokeStyle = "#c8c8c8";
    ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, drawW, drawH);
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
    args?.segments,
    localPicks,
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

  const cursorUV = (evt: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;
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
    const ox = (width - drawW) / 2;
    const oy = (height - drawH) / 2;
    const xClamped = Math.min(Math.max(x, ox), ox + drawW);
    const yClamped = Math.min(Math.max(y, oy), oy + drawH);
    const u = umin + (xClamped - ox) / scale;
    const v = vmax - (yClamped - oy) / scale;
    return { u, v };
  };

  const handleClick = (evt: React.MouseEvent<HTMLCanvasElement>) => {
    const { u, v } = cursorUV(evt);
    setLocalPicks((prev) => [...prev, [u, v]]);
    setPendingPicks((prev) => [...prev, [u, v]]);
    setCursor({ u, v });
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
        <button type="button" onClick={() => setZoom(1)}>
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
            setPendingPicks((prev) => prev.slice(0, -1));
          }}
          disabled={localPicks.length === 0}
        >
          Undo Last
        </button>{" "}
        zoom={zoom.toFixed(2)}x
      </div>
      <canvas
        ref={canvasRef}
        className="canvas"
        width={width}
        height={height}
        onMouseMove={(evt) => setCursor(cursorUV(evt))}
        onWheel={handleWheel}
        onClick={handleClick}
      />
      <div className="caption">
        Click to place markers, then press Commit Picks.
      </div>
      <div className="caption">{coordsText}</div>
    </div>
  );
};

const WrappedPlanePicker = withStreamlitConnection(PlanePicker);
export default WrappedPlanePicker;

const root = ReactDOM.createRoot(document.getElementById("root")!);
root.render(<WrappedPlanePicker />);
