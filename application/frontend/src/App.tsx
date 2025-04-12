// src/App.tsx
import { Routes, Route, useLocation } from "react-router-dom";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Configurations from "./pages/Configurations";
import Banner from "./components/Banner";
import Navbar from "./components/Navbar";

export default function App() {
  const location = useLocation();
  const isLoginPage = location.pathname === "/";

  return (
    <>
      <Banner />
      {!isLoginPage && <Navbar />}
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/configurations" element={<Configurations />} />
      </Routes>
    </>
  );
}
