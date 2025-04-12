// src/pages/Dashboard.tsx
import ReportGenerator from "../components/ReportGenerator";

export default function Dashboard() {

  return (
    <div className="min-h-screen bg-gray-50 p-6 flex justify-center">
      <div className="w-full max-w-2xl space-y-10">
        {/* Report Generator Section Only */}
        <ReportGenerator />
      </div>
    </div>
  );
}
