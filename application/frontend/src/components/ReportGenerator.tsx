import { useState } from "react";
import { Button } from "primereact/button";
import { Dialog } from "primereact/dialog";
import { ProgressSpinner } from "primereact/progressspinner";
import PanelCard from "./PanelCard";
import ConfigurationForm from "./ConfigurationForm";
import api from "../lib/api";

type Config = {
  id: number;
  area: number;
  plantation_cost: number;
  expected_yield: number;
  expected_profit: number;
};

type Props = {
  config: Config | null;
  onUpdateConfig: () => void;
  onReportGenerated: (result: any) => void;
};

export default function ReportGenerator({ config, onUpdateConfig, onReportGenerated }: Props) {
  const [showModal, setShowModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleGenerateReport = async () => {
    if (!config || !selectedFile) return;

    const formData = new FormData();
    formData.append("picture", selectedFile);

    try {
      setSubmitting(true);
      const response = await api.post(
        `/user/configurations/${config.id}/upload-photo`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      onReportGenerated(response.data);
    } catch (error) {
      console.error("Error uploading image:", error);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="p-4 space-y-6">
      <PanelCard title="Report Generator">
        {submitting ? (
          <div className="flex flex-col items-center justify-center h-64">
            <ProgressSpinner />
            <p className="mt-4 text-sm text-gray-600">Generating your report...</p>
          </div>
        ) : (
          <>
            {!config ? (
              <div className="flex flex-col justify-center p-3 my-4 shadow-lg rounded-md">
                <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 rounded-lg mb-6 ">
                  <p className="mb-2">
                    ⚠️ You don't have a configuration yet. Add one to continue.
                  </p>
                </div>
                <Button
                  label="Add Configuration"
                  icon="pi pi-cog"
                  onClick={() => setShowModal(true)}
                  className="p-button-sm self-end"
                />
              </div>
            ) : (
              <div className="text-sm text-gray-700 border border-gray-200 rounded-md p-4 mb-6 bg-gray-50">
                <h3 className="text-lg font-semibold mb-2">Saved configuration details</h3>
                <p><strong>📐 Area:</strong> {config.area} m²</p>
                <p><strong>💰 Plantation Cost:</strong> ${config.plantation_cost}</p>
                <p><strong>🌾 Expected Yield:</strong> {config.expected_yield}</p>
                <p><strong>💵 Expected Profit:</strong> ${config.expected_profit}</p>
              </div>
            )}

            {/* Upload Section */}
            <label className="w-full flex flex-col items-center justify-center border-2 border-dashed border-blue-400 rounded-xl px-4 py-8 cursor-pointer text-blue-600 hover:bg-blue-50 transition text-center">
              <i className="pi pi-upload text-4xl mb-2" />
              <span className="text-base font-medium">
                {selectedFile ? selectedFile.name : "Click here to upload an image"}
              </span>
              <input
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setSelectedFile(file);
                  }
                }}
              />
            </label>

            <div className="mt-6">
              <Button
                label="Generate the report"
                className="w-full p-button-success"
                disabled={!config || !selectedFile}
                onClick={handleGenerateReport}
              />
            </div>
          </>
        )}
      </PanelCard>

      {/* Configuration Modal */}
      <Dialog
        header="Add Configuration"
        visible={showModal}
        style={{ width: "500px" }}
        modal
        onHide={() => setShowModal(false)}
      >
        <ConfigurationForm
          onClose={() => setShowModal(false)}
          onSuccess={onUpdateConfig}
        />
      </Dialog>
    </div>
  );
}
