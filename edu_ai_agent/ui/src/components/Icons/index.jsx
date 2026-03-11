import Ai from '@/assets/icons/ai.svg?react';
import ArrowDown from '@/assets/icons/chevron-down.svg?react';
import ArrowUp from '@/assets/icons/chevron-up.svg?react';
import Copy from '@/assets/icons/copy.svg?react';
import Edit from '@/assets/icons/edit-2.svg?react';
import Home from '@/assets/icons/home.svg?react';
import HomeColor from '@/assets/icons/home_color.svg?react';
import Message from '@/assets/icons/message.svg?react';
import MessageColor from '@/assets/icons/message_color.svg?react';

import More from '@/assets/icons/more-vertical.svg?react';
import Plus from '@/assets/icons/plus.svg?react';
import Prompt from '@/assets/icons/prompt.svg?react';
import Search from '@/assets/icons/search.svg?react';
import Send from '@/assets/icons/send.svg?react';
import SendColor from '@/assets/icons/send_color.svg?react';
import Setting from '@/assets/icons/settings.svg?react';
import SettingColor from '@/assets/icons/settings_color.svg?react';
import SideBar from '@/assets/icons/sidebar.svg?react';
import Trash from '@/assets/icons/trash.svg?react';
import Close from '@/assets/icons/x.svg?react';

import Logo from '@/assets/icons/logo.svg?react';

const LogoIcon = ({ ...props }) => {
  return (<Logo width={40} height={40} fill={props.color} />);
};

const AiIcon = ({ ...props }) => {
  return (<Ai width={24} height={24} fill={props.color} />);
};

const ArrowDownIcon = ({ ...props }) => {
  return (<ArrowDown width={24} height={24} fill={props.color} />);
};

const ArrowUpIcon = ({ ...props }) => {
  return (<ArrowUp width={24} height={24} fill={props.color} />);
};

const CopyIcon = ({ ...props }) => {
  return (<Copy width={24} height={24} fill={props.color} />);
};

const EditIcon = ({ ...props }) => {
  return (<Edit width={24} height={24} fill={props.color} />);
};

const HomeIcon = ({ ...props }) => {
  return (<Home width={24} height={24} fill={props.color} />);
};
const HomeColorIcon = ({ ...props }) => {
  return (<HomeColor width={24} height={24} fill={props.color} />);
};

const MessageIcon = ({ ...props }) => {
  return (<Message width={24} height={24} fill={props.color} />);
};

const MessageColorIcon = ({ ...props }) => {
  return (<MessageColor width={24} height={24} fill={props.color} />);
};

const MoreIcon = ({ ...props }) => {
  return (<More width={24} height={24} fill={props.color} />);
};

const PlusIcon = ({ ...props }) => {
  return (<Plus width={24} height={24} fill={props.color} />);
};

const PromptIcon = ({ ...props }) => {
  return (<Prompt width={24} height={24} fill={props.color} />);
};

const SearchIcon = ({ ...props }) => {
  return (<Search width={24} height={24} fill={props.color} />);
};

const SendIcon = ({ ...props }) => {
  return (<Send width={24} height={24} fill={props.color} />);
};

const SendColorIcon = ({ ...props }) => {
  return (<SendColor width={24} height={24} fill={props.color} />);
};

const SettingIcon = ({ ...props }) => {
  return (<Setting width={24} height={24} fill={props.color} />);
};

const SettingColorIcon = ({ ...props }) => {
  return (<SettingColor width={24} height={24} fill={props.color} />);
};

const SideBarIcon = ({ ...props }) => {
  return (<SideBar width={24} height={24} fill={props.color} />);
};

const TrashIcon = ({ ...props }) => {
  return (<Trash width={24} height={24} fill={props.color} />);
};

const CloseIcon = ({ ...props }) => {
  return (<Close width={24} height={24} fill={props.color} />);
};

export { 
  LogoIcon,
  AiIcon, 
  ArrowDownIcon, 
  ArrowUpIcon, 
  HomeIcon,
  CopyIcon,
  EditIcon,
  MessageIcon,
  MoreIcon,
  PlusIcon,
  PromptIcon,
  SearchIcon,
  SendIcon,
  SettingIcon,
  SideBarIcon,
  TrashIcon,
  CloseIcon,
  HomeColorIcon,
  MessageColorIcon,
  SendColorIcon,
  SettingColorIcon
}