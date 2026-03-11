import { useEffect, useState } from 'react';
import { Button, CircularProgress, Container, Typography } from '@mui/material';
import { MessageInput } from '@/components/MessageInput';
import { useChat } from '@/hooks/useChat';
import styles from './ChatPage.module.scss'
import { CodeEditor } from '@/components/CodeEditor';
import { GridViewer } from '@/components/GridViewer';
import { ChartViewer } from '@/components/ChartViewer';

export const ChatPage = () => {
	const { container, wrapper, user, assistant, scroll, toolItems, content } = styles;

	const {
		answers,
		messages,
		question,
		messageEndRef,
		setMessages,
		scrollToBottom,
		handleSendMessage,
		handleSavePropmt
	} = useChat();

	useEffect(() => {
		const allMessages = [...question, ...answers].sort((a, b) =>
			new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
		);
		setMessages(allMessages);
	}, [question, answers]);


	useEffect(() => {
		const timer = setTimeout(() => {
			scrollToBottom();
		}, 500);
		return () => clearTimeout(timer);
	}, [messages]);

	const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

	const hasData = (obj: Object) => obj && Object.keys(obj).length > 0;
	// console.log(messages[1].metadata)
	return (
		<Container maxWidth={'lg'} className={container}>
			<div className={wrapper}>
				{messages?.map((msg, index) => (msg.role === 'user' ?
					<div
						key={index}
						className={user}
						onMouseEnter={() => setHoveredIndex(index)}
						onMouseLeave={() => setHoveredIndex(null)}>
						<Typography>
							{msg.content}
						</Typography>
						<Button onClick={() => handleSavePropmt(msg.content)} style={{
							// position: hoveredIndex === index ? 'relative' : 'absolute',
						}}>
							<Typography>
								{'Prompt 저장'}
							</Typography>
						</Button>
					</div> :
					<div key={index} className={assistant}>
						<div className={content}>
							{msg.isLoading && <CircularProgress size={24} />}
							<Typography>{msg.content}</Typography>
						</div>

						{msg.metadata &&
							<div className={toolItems}>
								{msg.metadata.sql &&
									<CodeEditor code={msg.metadata.sql} />
								}
								{hasData(msg.metadata?.data?.dataTable?.columns) &&
									<GridViewer data={msg.metadata.data} collapse={!hasData(msg.metadata?.chart?.chart_data)} />
								}
								{hasData(msg.metadata?.chart?.chart_data) &&
									<ChartViewer data={msg.metadata.chart?.chart_data} collapse={hasData(msg.metadata?.chart?.chart_data)} />
								}
							</div>
						}
					</div>
				))}

				<div className={scroll} ref={messageEndRef}></div>
			</div>
			<MessageInput handleSendMessage={handleSendMessage} />
		</Container>
	)
}