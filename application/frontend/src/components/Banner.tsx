export default function Banner() {
  return (
    <div className="relative w-full h-32 md:h-40 lg:h-48 overflow-hidden shadow-md">
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: 'url("/src/assets/wheat_tech.jpg")',
        }}
      />
      <div className="relative z-10 flex items-center justify-center h-full">
        <h1 className="text-white text-xl md:text-2xl lg:text-3xl font-bold drop-shadow-lg text-center">
          🌾 DeepWheat: AI-Powered Wheat Health Monitoring
        </h1>
      </div>
    </div>
  );
}
