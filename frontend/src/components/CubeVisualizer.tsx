import { useTrainingStore } from "../store/training";

// Colors for each face
const COLORS: Record<number, string> = {
  0: "#ffffff", // White (U)
  1: "#ff0000", // Red (R)
  2: "#00ff00", // Green (F)
  3: "#ffff00", // Yellow (D)
  4: "#ff8c00", // Orange (L)
  5: "#0000ff", // Blue (B)
};

// Face labels
const FACE_LABELS = ["U", "R", "F", "D", "L", "B"];

export function CubeVisualizer() {
  const { cubeState, scrambleMoves, lastMove, lastStepNum, lastStepSolved, lastStepDistance, evalMoveHistory, isTraining } = useTrainingStore();
  
  // Render 2D unfolded cube
  //        [U U U]
  //        [U U U]
  //        [U U U]
  // [L L L][F F F][R R R][B B B]
  // [L L L][F F F][R R R][B B B]
  // [L L L][F F F][R R R][B B B]
  //        [D D D]
  //        [D D D]
  //        [D D D]
  
  const renderFace = (faceStart: number, x: number, y: number) => {
    const stickers = [];
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const index = faceStart + row * 3 + col;
        const color = COLORS[cubeState[index]] || "#cccccc";
        stickers.push(
          <rect
            key={`${faceStart}-${row}-${col}`}
            x={x + col * 30}
            y={y + row * 30}
            width={28}
            height={28}
            fill={color}
            stroke="#333"
            strokeWidth={1}
            rx={2}
          />
        );
      }
    }
    return stickers;
  };
  
  return (
    <div className="cube-visualizer">
      <h3>Cube State</h3>
      <div className="cube-container">
        <svg width={360} height={360} viewBox="0 0 360 360">
          {/* U (White) - Top */}
          {renderFace(0, 90, 0)}
          
          {/* L (Orange) - Left */}
          {renderFace(36, 0, 90)}
          
          {/* F (Green) - Front */}
          {renderFace(18, 90, 90)}
          
          {/* R (Red) - Right */}
          {renderFace(9, 180, 90)}
          
          {/* B (Blue) - Back */}
          {renderFace(45, 270, 90)}
          
          {/* D (Yellow) - Bottom */}
          {renderFace(27, 90, 180)}
        </svg>
      </div>
      
      <div className="scramble-info">
        <h4>Scramble</h4>
        <p className="scramble-moves">
          {scrambleMoves.length > 0 
            ? scrambleMoves.join(" ")
            : "No scramble applied"
          }
        </p>
      </div>
      
      {isTraining && lastStepNum > 0 && (
        <div className="eval-step-info">
          <h4>Live Eval Step</h4>
          <div className="step-stats">
            <span className="step-stat">Step: <strong>{lastStepNum}</strong></span>
            <span className="step-stat">Move: <strong>{lastMove}</strong></span>
            <span className="step-stat">Distance: <strong>{lastStepDistance}</strong></span>
            <span className={`step-stat ${lastStepSolved ? "solved" : ""}`}>
              {lastStepSolved ? "✅ Solved!" : "🔄 Solving..."}
            </span>
          </div>
          {evalMoveHistory.length > 0 && (
            <div className="move-history">
              <span className="move-label">Moves:</span>
              <span className="move-sequence">
                {evalMoveHistory.map((m, i) => (
                  <span key={i} className={`move-chip ${i === evalMoveHistory.length - 1 ? "latest" : ""}`}>{m}</span>
                ))}
              </span>
            </div>
          )}
        </div>
      )}
      
      <div className="cube-legend">
        <h4>Faces</h4>
        <div className="legend-items">
          {FACE_LABELS.map((label, i) => (
            <div key={label} className="legend-item">
              <span
                className="legend-color"
                style={{ backgroundColor: COLORS[i] }}
              />
              <span>{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
