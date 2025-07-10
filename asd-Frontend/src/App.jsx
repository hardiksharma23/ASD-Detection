import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setResultImage(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an image file');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5000/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setResult(response.data.output);
      setResultImage(response.data.result_image);
    } catch (err) {
      setError('Error analyzing image: ' + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen h-screen w-screen bg-gradient-to-br from-gray-800 via-gray-700 to-gray-500 flex flex-col items-center overflow-y-auto p-6">
      <div className="flex flex-col items-center justify-start w-full max-w-md">
        {/* Header */}
        <h1 className="text-5xl font-extrabold text-green-100 mb-10 drop-shadow-lg">
          ASD Detection
        </h1>

        {/* Main Card */}
        <div className="bg-gray-600/50 backdrop-blur-md shadow-lg rounded-xl p-6 w-full">
          {/* File Input and Button */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Upload Image
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-600 transition-colors duration-300"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className={`w-full py-3 px-4 rounded-lg text-red-500 font-semibold bg-green-200 hover:bg-green-100 ${
                loading ? 'opacity-70 cursor-not-allowed' : ''
              } transition-colors duration-300 shadow-md`}
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          </form>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-3 bg-red-500/20 backdrop-blur-md text-red-100 rounded-lg border border-red-500/50">
              {error}
            </div>
          )}
        </div>

        {/* Result Section */}
        {result && (
          <div className="mt-8 w-full">
            <h2 className="text-3xl font-semibold text-white mb-4 drop-shadow-md">
              Result
            </h2>
            <div className="bg-gray-600/50 backdrop-blur-md shadow-lg rounded-xl p-6">
              <p className="text-lg font-medium text-gray-100 mb-4 capitalize">
                {result}
              </p>
              {resultImage && (
                <div className="mt-4">
                  <h3 className="text-lg font-medium text-gray-100 mb-2">Result Image</h3>
                  <div className="overflow-hidden rounded-lg">
                    <img
                      src={`data:image/jpeg;base64,${resultImage}`} // Base64-encoded image
                      alt="Result"
                      className="w-full h-auto max-h-64 object-contain rounded-lg shadow-md"
                      onError={(e) => console.error("Failed to load image:", e)}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;