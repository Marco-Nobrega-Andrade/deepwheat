import { useState } from "react";
import { Button } from "primereact/button";
import api from "../lib/api";

type Config = {
  size: number;     // maps to area
  cost: number;     // maps to plantation_cost
  price: number;    // maps to expected_yield
  profit: number;   // maps to expected_profit
};

type Props = {
  onCloseModal?: () => void;
};

export default function ConfigurationForm({ onCloseModal }: Props) {
  const [submitted, setSubmitted] = useState(false);
  const [config, setConfig] = useState<Config>({
    size: 0,
    cost: 0,
    price: 0,
    profit: 0,
  });

  const isFormValid =
    config.size > 0 &&
    config.cost > 0 &&
    config.price > 0 &&
    config.profit > 0;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);

    if (!isFormValid) return;

    try {
      const response = await api.post("/user/configurations", {
        area: config.size,
        plantation_cost: config.cost,
        expected_yield: config.price,
        expected_profit: config.profit,
      });

      console.log("Configuration saved:", response.data);
      onCloseModal?.();
    } catch (error) {
      console.error("API error:", error);
      // Optionally: show toast or user-friendly error message
    }
  };

  const handleChange = (field: keyof Config, value: number) => {
    setConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block mb-1 text-sm font-medium text-gray-600">
          Plantation Size (m²)
        </label>
        <input
          type="number"
          value={config.size}
          onChange={(e) => handleChange("size", Number(e.target.value))}
          className="w-full px-4 py-2 border rounded-xl focus:ring focus:ring-blue-300"
        />
        {submitted && config.size <= 0 && (
          <small className="text-red-500">Size is required</small>
        )}
      </div>

      <div>
        <label className="block mb-1 text-sm font-medium text-gray-600">
          Cost per m²
        </label>
        <input
          type="number"
          value={config.cost}
          onChange={(e) => handleChange("cost", Number(e.target.value))}
          className="w-full px-4 py-2 border rounded-xl focus:ring focus:ring-blue-300"
        />
        {submitted && config.cost <= 0 && (
          <small className="text-red-500">Cost is required</small>
        )}
      </div>

      <div>
        <label className="block mb-1 text-sm font-medium text-gray-600">
          Expected Yield
        </label>
        <input
          type="number"
          value={config.price}
          onChange={(e) => handleChange("price", Number(e.target.value))}
          className="w-full px-4 py-2 border rounded-xl focus:ring focus:ring-blue-300"
        />
        {submitted && config.price <= 0 && (
          <small className="text-red-500">Expected yield is required</small>
        )}
      </div>

      <div>
        <label className="block mb-1 text-sm font-medium text-gray-600">
          Expected Profit
        </label>
        <input
          type="number"
          value={config.profit}
          onChange={(e) => handleChange("profit", Number(e.target.value))}
          className="w-full px-4 py-2 border rounded-xl focus:ring focus:ring-blue-300"
        />
        {submitted && config.profit <= 0 && (
          <small className="text-red-500">Expected profit is required</small>
        )}
      </div>

      <div className="mt-6 flex justify-end">
        <Button
          type="submit"
          label="Save Configuration"
          icon="pi pi-save"
          className="p-button-primary"
          disabled={!isFormValid}
        />
      </div>
    </form>
  );
}
