import { ReportResult } from "../models/report-result.model";

// src/components/ReportResultPanel.tsx
type Props = {
  result: ReportResult;
};

export default function ReportResultPanel({ result }: Props) {
  const isHealthy = result.wheat_disease === "Healthy";

  return (
    <div className="bg-white border border-gray-200 shadow-md rounded-xl p-6 space-y-6">
      <h2 className="text-2xl font-bold text-blue-700">
        🧬 Wheat Health Report
      </h2>

      {isHealthy ? (
        <div className="bg-green-50 border-l-4 border-green-500 text-green-800 p-4 rounded-lg">
          <h3 className="text-xl font-semibold mb-1">✅ No Disease Detected</h3>
          <p className="text-sm">Your wheat crop appears to be healthy and free from disease. Keep monitoring regularly for early detection!</p>
        </div>
      ) : (
        <>
          <h3 className="text-lg font-semibold text-gray-700">
            🦠 Detected Disease: {result.wheat_disease}
          </h3>

          <div>
            <h4 className="text-md font-medium text-gray-700 mb-2">📊 Infection Timeline</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm text-left border">
                <thead className="bg-blue-50 text-blue-700">
                  <tr>
                    <th className="px-4 py-2 border">Day</th>
                    <th className="px-4 py-2 border">Infected Area</th>
                    <th className="px-4 py-2 border">Yield Loss</th>
                    <th className="px-4 py-2 border">Profit Loss</th>
                    <th className="px-4 py-2 border">Plantation Loss</th>
                  </tr>
                </thead>
                <tbody>
                  {result.infection_by_day.map((entry, index) => (
                    <tr key={index} className="even:bg-gray-50">
                      <td className="px-4 py-2 border">{entry.day}</td>
                      <td className="px-4 py-2 border">{entry.infected_area}</td>
                      <td className="px-4 py-2 border">{entry.yield_loss}</td>
                      <td className="px-4 py-2 border">{entry.economic_impact.profit_loss}</td>
                      <td className="px-4 py-2 border">{entry.economic_impact.plantation_loss}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-yellow-50 border-l-4 border-yellow-400 text-yellow-800 p-4 rounded-lg">
            <strong>⚠️ Treatment Info:</strong> {result.disease_treatment}
          </div>
        </>
      )}
    </div>
  );
}
