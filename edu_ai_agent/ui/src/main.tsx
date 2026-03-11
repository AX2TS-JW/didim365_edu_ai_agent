import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import GlobalStyles from '@/assets/GlobalStyles';
import QueryProvider from '@/components/providers/QueryProvider';
import { createTheme, ThemeProvider } from '@mui/material';
import { Provider } from 'jotai';
import './constants/i18n/config';

const theme = createTheme({
  typography: {
    fontFamily: [
      'Pretendard'
    ].join(','),
  },
  palette: {
    primary: {
      main: '#344460'
    },
    secondary: {
      main: '#253248',
    }
  }
});

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <QueryProvider>
        <GlobalStyles />
        <Provider>
        <App />
        </Provider>
      </QueryProvider>
    </ThemeProvider>
  </React.StrictMode>,
);
