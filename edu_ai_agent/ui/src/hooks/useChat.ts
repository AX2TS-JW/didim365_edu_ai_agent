import { ChatService } from "@/services/chatService";
import { answerAtom } from "@/store/answer";
import { messageAtom, threadById } from "@/store/message";
import { isPromptModalAtom, promptAtom } from "@/store/prompts";
import { questionAtom } from "@/store/question";
import { IMessage } from "@/types/chatVM";
import { useAtom } from "jotai";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { v4 as uuidv4 } from 'uuid';

export const useChat = () => {
  const chatService = ChatService();
  const navigate = useNavigate();
  const [answers, setAnswer] = useAtom(answerAtom);
  const [question, setQuestion] = useAtom(questionAtom);
  const [prompt, setPrompt] = useAtom(promptAtom);
  const [threadId, setThreadId] = useAtom(threadById);

  const [messages, setMessages] = useState<IMessage[]>([])
  const [message, setMessage] = useAtom(messageAtom);
  const [isTyping, setIsTyping] = useState(false);

  const [isOpen, setIsOpen] = useAtom(isPromptModalAtom);

  const messageEndRef = useRef<HTMLDivElement>(null);

  const openPromptModal = () => {
    setIsOpen(!isOpen);
  }

  useEffect(() => {
    setPrompt([
      'Agent에게 인사하기',
      '간단한 업무 요청하기',
      '주요 뉴스 요약 요청하기'
    ])
  }, []);

  const handleSavePropmt = (prompt: string) => {
    setPrompt(prev => [...prev, prompt]);
  };

  const scrollToBottom = () => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleChunk = (step: string, content: string, metadata, toolCalls: string[], name: string) => {
    setAnswer(prev => {
      const newAnswers = [...prev];
      if (newAnswers.length > 0) {
        newAnswers[newAnswers.length - 1] = {
          ...newAnswers[newAnswers.length - 1],
          content: step === 'done' ? content
            : (toolCalls && toolCalls.length > 0 ? `${toolCalls.join(', ')} 이해하고 있습니다.` :
              `${name.replace(/.*?: /, '')} 처리 중입니다.`),
          isLoading: step !== 'done',
          metadata: metadata ?? null
        };
      }
      return newAnswers;
    });

    if (step === 'done') {
      setIsTyping(false);
      return
    }
  }

  const handleSendMessage = (inputValue: string) => {
    if (isTyping) return; // 중복 방지
    // 사용자 질문
    setQuestion(prev => [...prev, {
      message_id: uuidv4(),
      role: 'user',
      content: inputValue,
      likeStatus: 0,
      createdAt: new Date()
    }])

    setIsTyping(true);

    const assistantMessage = {
      messageId: uuidv4(),
      role: 'assistant',
      content: '',
      likeStatus: 0,
      isFavorited: false,
      createdAt: new Date(),
    }

    setAnswer(prev => [...prev, assistantMessage]);
    const isNewChat = !threadId;

    const currentThreadId = isNewChat ? uuidv4() : threadId;

    if (isNewChat) {
      setThreadId(currentThreadId);
    };

    // AI 답변 
    chatService.sendMessage(
      currentThreadId,
      inputValue,
      handleChunk
    );

    if (location.pathname !== '/chat') {
      navigate('/chat');
    }
    setMessage('');
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // 기본 줄바꿈 동작 방지
      handleSendMessage(message); // 메시지 전송 함수 호출
    }
  }

  const handleMsgChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
  }

  const handleUsePrompt = (prompt: string) => {
    setIsOpen(false);
    setMessage(prompt);
  }

  return {
    answers,
    chatService,
    isOpen,
    isTyping,
    message,
    messages,
    question,
    messageEndRef,
    handleSendMessage,
    setAnswer,
    setIsOpen,
    setIsTyping,
    setPrompt,
    setMessages,
    setQuestion,
    setThreadId,
    scrollToBottom,
    onKeyDown,
    handleMsgChange,
    setMessage,
    openPromptModal,
    handleSavePropmt,
    handleUsePrompt,
  }
}
