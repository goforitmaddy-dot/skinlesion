interface Props {
  prediction: string;
  confidence: number;
}

export default function ResultCard({ prediction, confidence }: Props) {
  const percentage = Math.round(confidence * 100);

  const confidenceColor =
    percentage > 80
      ? "bg-green-500"
      : percentage > 50
      ? "bg-yellow-500"
      : "bg-red-500";

  return (
    <div className="bg-gray-50 border rounded-xl p-5 space-y-4 animate-in fade-in duration-500">
      <div>
        <h2 className="font-semibold text-lg">Prediction Result</h2>

        <p className="text-sm text-gray-500">AI-generated analysis</p>
      </div>

      <div>
        <p className="font-medium">Prediction</p>

        <p className="text-2xl font-bold capitalize">{prediction}</p>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Confidence</span>
          <span>{percentage}%</span>
        </div>

        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <div
            className={`${confidenceColor} h-3 rounded-full transition-all duration-700`}
            style={{
              width: `${percentage}%`,
            }}
          />
        </div>
      </div>

      <div className="text-xs text-gray-500 border-t pt-3">
        This AI prediction is not a medical diagnosis. Please consult a
        dermatologist.
      </div>
    </div>
  );
}
