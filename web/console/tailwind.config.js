/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"DM Sans"', '-apple-system', 'sans-serif'],
        mono: ['"DM Mono"', 'monospace'],
      },
      colors: {
        canvas: '#F3F7F6', // page background
        surface: '#FFFFFF', // cards, header
        ink: '#10201F', // primary text, dark cards
        muted: '#3F5654', // secondary labels
        teal: {
          DEFAULT: '#0B6B66', // primary actions, links, active tabs
          dark: '#064440', // hover
        },
        subtle: '#EAF1F0', // card backgrounds, search pill
        border: '#D5E0DE',
        field: '#AFC3C0', // input borders
        chip: '#DDEBE9', // tags, badges
        amber: '#B7791F', // warnings
        red: '#A1332C', // errors
      },
      keyframes: {
        'toast-in': {
          '0%': { opacity: '0', transform: 'translateY(-8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'toast-in': 'toast-in 200ms ease-out',
      },
    },
  },
  plugins: [],
};
