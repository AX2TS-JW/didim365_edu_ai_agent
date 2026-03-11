export type sqlSchema = {

}
export type dataTable = {
	columns: {}
}
export type gridSchema = {
	dataTable: dataTable
}

export type chartSchema = {
	chart_data: Object
}

export type metaSchema = {
	sql: string
	data: gridSchema,
	chart: chartSchema, 
}

export interface ChatResponse {
  step: string;
	name: string;
	toolCalls: string[];
	role?: string;
	content?: string; 
	metadata?: metaSchema;
	type: string;
}

export interface IMessage {
	createdAt: Date;
	isFavorited: boolean;
	likeStatus: number;
	content: string;
	messageId: string;
	role: string;
	metadata?: metaSchema;
	isLoading?: boolean;
}