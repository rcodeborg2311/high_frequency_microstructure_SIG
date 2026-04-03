/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:      '#060a14',
        surface: '#0d1321',
        card:    '#0f1729',
        border:  '#1a2744',
        accent:  '#0088ff',
        cyan:    '#00d4ff',
        green:   '#00c853',
        red:     '#ff3b3b',
        yellow:  '#ffab00',
        text:    '#e2e8f0',
        muted:   '#4a5568',
        dim:     '#2d3748',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        glow:        '0 0 20px rgba(0,136,255,0.15)',
        'glow-green':'0 0 20px rgba(0,200,83,0.15)',
        'glow-red':  '0 0 20px rgba(255,59,59,0.15)',
      },
    },
  },
  plugins: [],
}
