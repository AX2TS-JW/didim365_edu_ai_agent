import { convertKeysToCamelCase } from '@/utils/utils';
import axios from 'axios';
import type { AxiosResponse } from 'axios';

/**
 * Request Success Handler
 */
const requestSuccessHandler = config => {
  return config;
};

/**
 * Request Fail Handler
 */
const requestErrorHandler = err => {
  return Promise.reject(err);
};

/**
 * Response Success Handler
 */
const responseSuccessHandler = res => {
  const response: AxiosResponse = res.data;

  if (200 <= res.status && res.status < 300) {
    return response.data;
  } else {
    return responseErrorHandler(res);
  }
};

/**
 * Response Fail handler
 */
const responseErrorHandler = err => {
  return Promise.reject(err);
};

/**
 * Axios 객체
 */
const axiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL + '/api/',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

/**
 * Axios Request Middleware
 */
// axiosInstance.interceptors.request.use(
//   config => requestSuccessHandler(config),
//   err => requestErrorHandler(err),
// );

// /**
//  * Axios Response Middleware
//  */
axiosInstance.interceptors.response.use(
  (response) => {
    if(response.data && Array.isArray(response.data.response)) {
      response.data.response = response.data.response.map(item =>
        convertKeysToCamelCase(item)
      );
    }
    // 수정 필요!
    if(response.data && Array.isArray(response.data.response.messages)){
      response.data.response.messages = response.data.response.messages.map(item =>
        convertKeysToCamelCase(item)
      )
    }
    return response;
  },
  (error) => Promise.reject(error)
);

export default axiosInstance;
