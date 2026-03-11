import api from './common'

export const ChatService = () => ({
  sendMessage: (threadId, message: string, onChunk) => {
    try {
      const baseURL = import.meta.env.VITE_API_BASE_URL;
      // const baseURL = 'http://localhost:5000'; 
      // baseURL + '/stream/chat'
      // return api.stream(baseURL + '/stream/chat', { "thread_id": threadId, message: message }, onChunk);

      return api.stream(baseURL + '/api/v1/chat', { "thread_id": threadId, message: message }, onChunk);
    } catch (error) {
      console.error('Failed to send message:', error)
      throw error;
    }
  },

  getFavoriteList: async() => {
    try {
      return await api.get('/v1/favorites/questions');
    } catch (error) {
      console.error('Failed to get favorite list:', error)
    }
  },

  getThreads: async() => {
    try {
      return await api.get('/v1/threads');
    } catch (error) {
      console.error('Failed to get threads:', error);
      throw error;
    }
  },

  getThreadById: async({ threadId }) => {
    try {
      return await api.get('/v1/threads/' + threadId);
    } catch (error) {
      console.error('Failed to get thread by id:', error);
      throw error;
    }
  }
})