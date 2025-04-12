// src/components/ReportGenerator.tsx
import { useState } from "react";
import { Button } from "primereact/button";
import { Dialog } from "primereact/dialog";
import PanelCard from "./PanelCard";
import ConfigurationForm from "./ConfigurationForm";

export default function ReportGenerator() {
  const [showModal, setShowModal] = useState(false);

  return (
    <div className="p-4 space-y-6">
      <PanelCard title="Report Generator">
        <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 rounded-lg mb-6">
          <Button
            label="Add Configuration"
            icon="pi pi-cog"
            onClick={() => setShowModal(true)}
            className="p-button-sm"
          />
        </div>

        {/* Upload full-width */}
        <label className="w-full flex flex-col items-center justify-center border-2 border-dashed border-blue-400 rounded-xl px-4 py-8 cursor-pointer text-blue-600 hover:bg-blue-50 transition text-center">
          <i className="pi pi-upload text-4xl mb-2" />
          <span className="text-base font-medium">Click here to upload an image</span>
          <input
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              if (e.target.files?.[0]) {
                alert(`Uploaded: ${e.target.files[0].name}`);
              }
            }}
          />
        </label>

        <div className="mt-6">
          <Button
            label="Generate the report"
            icon="pi pi-search"
            className="w-full p-button-success"
            onClick={() => alert("Image sent to analysis (mocked)")}
          />
        </div>
      </PanelCard>

      {/* Modal with ConfigurationForm */}
      <Dialog
        header="Add Configuration"
        visible={showModal}
        style={{ width: '500px' }}
        modal
        onHide={() => setShowModal(false)}
      >
        <ConfigurationForm onCloseModal={() => setShowModal(false)} />
      </Dialog>
    </div>
  );
}
