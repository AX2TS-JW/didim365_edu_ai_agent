import { Layout } from "@/components/Layout";
import { ChatPage } from "@/pages/ChatPage";
import InitPage from "@/pages/InitPage";
import { RouteObject } from "react-router-dom";

type RouteType = {
	id: string;
	path: string;
	label: string;
	element?: React.ReactNode;
	children?: RouteType[];
} & Omit<RouteObject, 'children'>;

export const routeConfig: RouteType[] = [
	{
		id: 'layout',
		path: '/',
		label: 'layout',
		element: <Layout />,
		children: [
			{
				id: 'InitPage',
				path: '/',
				label: 'InitPage',
				element: <InitPage />
			},
			{
				id: 'ChatPage',
				path: 'chat',
				label: 'ChatPage',
				element: <ChatPage />
			},
			{
				id: 'Dashboard',
				path: 'dashboard',
				label: 'Dashboard',
				element: <></>
			},
			{
				id: 'Setting',
				path: 'setting',
				label: 'Setting',
				element: <></>
			}
		]
	}
]