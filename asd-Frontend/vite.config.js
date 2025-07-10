import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react'; // Add React plugin
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [
    react(), // Add this for React support
    tailwindcss(),
  ],
  build: {
    outDir: 'dist', // Ensure output directory is 'dist'
    assetsDir: 'assets', // Ensure assets are placed in 'assets' folder
  },
  base: '/', // Ensure base path is root for Amplify
});