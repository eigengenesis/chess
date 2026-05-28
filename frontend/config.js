(() => {
  const host = window.location.hostname;
  const local = host === "" || host === "localhost" || host === "127.0.0.1";

  window.CHESS_GOD_CONFIG = {
    apiBaseUrl: local ? "http://127.0.0.1:8000/api" : "/api",
  };
})();
