import { fetchEventSource } from "@microsoft/fetch-event-source";
import axiosInstance from ".";
import { convertKeysToCamelCase } from "@/utils/utils";
import { ChatResponse } from "@/types/chatVM";


const api = {
  // GET 요청
  get: async (url, config = {}) => {
    try {
      const response = await axiosInstance.get(url, config)
    
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // POST 요청
  post: async (url, data = {}, config = {}) => {
    try {
      const response = await axiosInstance.post(url, data, config);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // PUT 요청
  put: async (url, data = {}, config = {}) => {
    try {
      const response = await axiosInstance.put(url, data, config);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // PATCH 요청
  patch: async (url, data = {}, config = {}) => {
    try {
      const response = await axiosInstance.patch(url, data, config);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // DELETE 요청
  delete: async (url, config = {}) => {
    try {
      const response = await axiosInstance.delete(url, config);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  stream: async(url, data = {}, onChunk) => {
    try{
      const controller = new AbortController();
      await fetchEventSource(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(data),
        signal: controller.signal,  // AbortController 연결
        openWhenHidden: true,       // 탭 전환 시 재연결 방지

        async onopen(res) {
          if(!res.ok){
            controller.abort();
          }
        },

        onmessage(event) {
          try {
            const data = JSON.parse(event.data);

            const { step, toolCalls, content, metadata, name } = convertKeysToCamelCase(data) as ChatResponse;

            onChunk(step, content, metadata, toolCalls, name);

            // done 이벤트 수신 시 SSE 연결 종료 (재연결 방지)
            if(step === 'done') {
              controller.abort();
            }
          } catch(err) {
            console.log(err);
          }
        },
        onclose() {
          // 서버가 연결을 닫으면 재연결하지 않음
          controller.abort();
        },
        onerror(err){
          console.log(err);
          controller.abort();
          throw err;  // 자동 재연결 완전 중단
        }
      })
    } catch(err) {
      // AbortError는 정상 종료이므로 무시
      if(err instanceof DOMException && err.name === 'AbortError') return;
      console.log(err)
    }
  } 
}

export default api;
