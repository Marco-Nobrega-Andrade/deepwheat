// src/components/ReportResultPanel.tsx
type Props = {
  result: any; // use a proper type when your report structure is defined
};

export default function ReportResultPanel({ result }: Props) {
  return (
    <div className="w-full md:w-[300px] lg:w-[350px] p-4 bg-white rounded-xl shadow space-y-4 border border-gray-200">
      <h2 className="text-lg font-semibold text-gray-800">📊 Report Summary</h2>

      <div className="text-sm text-gray-700 space-y-2">
        <p><strong>Status:</strong> {result.status || "Pending"}</p>
        <p><strong>Detected Disease:</strong> {result.disease || "N/A"}</p>
        <p><strong>Confidence:</strong> {result.confidence ? `${result.confidence}%` : "N/A"}</p>
        {/* Add more based on your actual response */}
      </div>
    </div>
  );
}
