import { useEffect, useState } from 'react';
import { Card, CardContent, Container, Grid2, Typography } from '@mui/material';
import { AiIcon } from '@/components/Icons';
import { MessageInput } from '@/components/MessageInput';
import { useChat } from '@/hooks/useChat';

import styles from './InitPage.module.scss';
import { useHistory } from '@/hooks/useHistory';
type FavoriteList = {
  createAt: string;
  questionId: string;
  title: string;
  type: string;
  updatedAt: string;
}

const InitPage = () => {
  const { container, wrapper, grid, cardGrid, cardContent } = styles;
  const [cards, setCards] = useState<FavoriteList[]>([]);

  const {
    chatService,
    handleSendMessage,
    setQuestion,
    setAnswer,
    setPrompt,
    setThreadId
  } = useChat();

  const {
    loadHistory
  } = useHistory();

  useEffect(() => {
    setQuestion([]);
    setAnswer([]);
    setPrompt([
      'Agent에게 인사하기',
      '간단한 업무 요청하기',
      '주요 뉴스 요약 요청하기'
    ]);
    setThreadId('');
  }, []);

  useEffect(() => {
    const fetchFavorite = async () => {
      try {
        const mainColList = await chatService.getFavoriteList();
        if (mainColList.response) {
          setCards(mainColList.response as FavoriteList[]);
        }
      } catch (error) {
        console.error('Failed to load favorites:', error);
        setCards([]);
      }
    };
    fetchFavorite();
  }, []);

  return (
    <Container maxWidth={'lg'} className={container}>
      <div className={wrapper}>
        <h3>
          <em>{'Agent Template'}</em>
          {'에 오신 것을 환영합니다'}
        </h3>
        <h5>{'Agent 교육을 위한 챗봇 프론트엔드 환경입니다.'}</h5>
      </div>

      <div className={cardGrid}>
        <Typography
          variant="h6"
        >
          <AiIcon />
          <span>{'아래 작업을 할 수 있어요'}</span>
        </Typography>

        <Grid2 container spacing={2} className={grid}>
          {cards?.map((card, index) => (
            <Grid2 size={{ xs: 12, sm: 6, md: 4 }} key={index} >
              <Card sx={{
                borderRadius: '8px',
                boxShadow: '0 0 12px rgba(0, 0, 0, 0.08)',
                '&:hover': {
                  boxShadow: '0 0 12px 0 rgba(123, 105, 241, 0.40)',

                }
              }}>
                <CardContent className={cardContent} onClick={() => handleSendMessage(card.title)}>
                  <Typography
                    variant='subtitle2'
                  >
                    {card.type}
                  </Typography>
                  <Typography
                    variant='body2'
                  >
                    {card.title}
                  </Typography>
                </CardContent>
              </Card>
            </Grid2>
          ))}
        </Grid2>
      </div>

      <MessageInput handleSendMessage={handleSendMessage} />
    </Container>
  )
}

export default InitPage;