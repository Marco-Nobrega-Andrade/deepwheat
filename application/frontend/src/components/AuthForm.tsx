import { useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../lib/api"; // make sure axios is set up here

export default function AuthForm() {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const payload =
        mode === "login"
          ? { email, password }
          : { email, password, name };

      const response = await api.post(
        mode === "login" ? "/login" : "/register",
        payload
      );

      const token = response.data.token;
      localStorage.setItem("token", token);

      navigate("/dashboard");
    } catch (err: any) {
      const message = err.response?.data?.message || "Something went wrong";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-white p-8 rounded-2xl shadow-md w-full max-w-md space-y-6"
    >
      <h2 className="text-2xl font-bold text-center text-gray-800">
        {mode === "login" ? "Login" : "Register"}
      </h2>

      {error && (
        <p className="text-center text-sm text-red-600 font-medium">{error}</p>
      )}

      {mode === "register" && (
        <div>
          <label className="block text-sm font-medium text-gray-600 mb-1">
            Name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-gray-600 mb-1">
          Email
        </label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-600 mb-1">
          Password
        </label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <button
        type="submit"
        className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-xl transition disabled:opacity-50"
        disabled={loading}
      >
        {loading
          ? mode === "login"
            ? "Signing In..."
            : "Registering..."
          : mode === "login"
            ? "Sign In"
            : "Register"}
      </button>

      <p className="text-center text-sm text-gray-600">
        {mode === "login" ? "Don't have an account?" : "Already have an account?"}{" "}
        <button
          type="button"
          onClick={() => setMode(mode === "login" ? "register" : "login")}
          className="text-blue-600 hover:underline font-medium ml-1"
        >
          {mode === "login" ? "Register" : "Login"}
        </button>
      </p>
    </form>
  );
}
