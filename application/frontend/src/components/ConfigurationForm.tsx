import { useEffect, useState } from "react";
import { Button } from "primereact/button";
import api from "../lib/api";

type Config = {
  id?: number;
  area: number;
  plantation_cost: number;
  expected_yield: number;
  expected_profit: number;
};

type Props = {
  initialConfig?: Config;
  onClose: () => void;
  onSuccess?: () => void;
};

export default function ConfigurationForm({ initialConfig, onClose, onSuccess }: Props) {
  const [config, setConfig] = useState<Config>({
    area: NaN,
    plantation_cost: NaN,
    expected_yield: NaN,
    expected_profit: NaN,
  });


  const [submitted, setSubmitted] = useState(false);
  const isFormValid =
    config.area > 0 &&
    config.plantation_cost > 0 &&
    config.expected_yield > 0 &&
    config.expected_profit > 0;

  useEffect(() => {
    if (initialConfig) {
      setConfig(initialConfig);
    }
  }, [initialConfig]);

  const handleChange = (field: keyof Config, value: number) => {
    setConfig((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    if (!isFormValid) return;

    try {
      if (config.id) {
        await api.put(`/user/configurations/${config.id}`, config);
      } else {
        await api.post("/user/configurations", config);
      }
      onSuccess?.();
      onClose();
    } catch (error) {
      console.error("Error submitting configuration:", error);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {["area", "plantation_cost", "expected_yield", "expected_profit"].map((field) => (
        <div key={field}>
          <label className="block mb-1 text-sm font-medium text-gray-600 capitalize">
            {field.replace(/_/g, " ")}
          </label>
          <input
            type="number"
            min="0"
            value={config[field as keyof Config] || ""}
            onChange={(e) => {
              const value = Number(e.target.value);
              handleChange(field as keyof Config, value < 0 ? 0 : value); // prevent negatives
            }}
            className="w-full px-4 py-2 border rounded-xl focus:ring focus:ring-blue-300 appearance-none [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
          />



          {submitted && (config[field as keyof Config] ?? 0) <= 0 && (
            <small className="text-red-500">This field is required</small>
          )}
        </div>
      ))}

      <div className="mt-6 flex justify-end">
        <Button
          type="submit"
          label={config.id ? "Update" : "Save"}
          icon="pi pi-save"
          className="p-button-primary"
          disabled={!isFormValid}
        />
      </div>
    </form>
  );
}
