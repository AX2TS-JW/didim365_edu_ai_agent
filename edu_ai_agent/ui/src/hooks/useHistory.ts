import { ChatService } from "@/services/chatService";
import { answerAtom } from "@/store/answer";
import { threadById } from "@/store/message";
import { questionAtom } from "@/store/question";
import { useSetAtom } from "jotai";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

type threadType = {
  createAt: string;
  isFavorited: boolean;
  threadId: string;
  title: string;
  type: string;
  updatedAt: string;
}

export const useHistory = () => {
  const chatService = ChatService();
  const navigate = useNavigate();
  
  const setAnswer = useSetAtom(answerAtom);
  const setQuestion = useSetAtom(questionAtom);
  const setThreadId = useSetAtom(threadById);

  const [bookmark, setBookmark] = useState<threadType[]>([])
  const [histories, setHisories] = useState<threadType[]>([]);
  const [searchResults, setSearchResults] = useState<threadType[]>([])

  const [keyword, setKeyword] = useState('');
  const [collapsed, setCollapsed] = useState(true);

  useEffect(()=>{
    chatService.getThreads().then((res)=>{
      const bookmarked = res.response.filter(item => item.isFavorited === true);
      const historyList = res.response.filter(item => item.isFavorited === false);
      setBookmark(bookmarked);
      setHisories(historyList);
      setSearchResults(historyList);
    })
  }, []);

  const searchByTitle = (data, keyword, caseSensitive = false) => {
    if(!keyword || keyword.trim() === ''){
      return data;
    }

    const searchTerm = caseSensitive ? keyword : keyword.toLowerCase();

    return data.filter(item => {
      const title = caseSensitive ? item.title : item.title.toLowerCase();
      return title.includes(searchTerm);
    })
  }

  const loadHistory = (threadId: string) => {
    chatService.getThreadById({ threadId }).then((res)=> {
      const userMessage = res.response.messages.filter(item => item.role === 'user');
      const assistantMessage = res.response.messages.filter(item => item.role === 'assistant');
      setAnswer(assistantMessage);
      setQuestion(userMessage);
      setThreadId(res.response.thread_id);
      navigate('/chat')
    })
  };

  const handleSearch = (e) => {
    const value = e.target.value;
    const results = searchByTitle(histories, value);
    setKeyword(value);
    setSearchResults(results);
  };

  const handleDrawerToggle = () => {
    setCollapsed(!collapsed);
  };

  return {
    collapsed,
    bookmark,
    searchResults,
    keyword,
    setAnswer,
    setQuestion,
    loadHistory,
    handleSearch,
    navigate,
    handleDrawerToggle
  }
}