import { Box, Button, Divider, Drawer, InputAdornment, TextField, Typography } from '@mui/material';
import styles from './SubMenu.module.scss'

import { PlusIcon, SearchIcon, SideBarIcon } from '@/components/Icons';
import { useState } from 'react';
import { useHistory } from '@/hooks/useHistory';

const drawerWidth = 280;



export const SubMenu = () => {
  const {
    collapsed,
    keyword,
    bookmark,
    searchResults,
    navigate,
    loadHistory,
    handleSearch,
    handleDrawerToggle
  } = useHistory();

  const { container, header, searchInput, subMenu, sidebarIcon } = styles;

  return (
    <>{collapsed ?
      <Drawer
        variant="permanent"
        className={container}
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            left: 64,
            bgcolor: '#F5F5F8',
            color: 'white',
            borderRight: '1px solid #E0E4E5',
            //boxShadow: '0 0px 2px 0 rgba(0, 0, 0, 0.05)), 0 4px 8px 0 rgba(0, 0, 0, 0.08));'
          },
        }}
      >
        <div className={header}>
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'flex-end',
              padding: '14px 0'
            }}
          >
            <i onClick={handleDrawerToggle}>
              <SideBarIcon></SideBarIcon>
            </i>
          </Box>

          <Button variant='contained' onClick={() => navigate('/')}>
            <PlusIcon color={'white'} />
            <Typography
              sx={{
                fontSize: '16px !important'
              }}
            >
              {'새 대화'}
            </Typography>
          </Button>

          <Box sx={{ margin: '12px 0 24px 0' }}>
            <TextField
              placeholder='대화 검색'
              className={searchInput}
              value={keyword}
              onChange={handleSearch}
              slotProps={{
                input: {
                  endAdornment: (
                    <InputAdornment position='end'>
                      <SearchIcon />
                    </InputAdornment>
                  )
                }
              }}
            />
          </Box>
        </div>

        <div className={subMenu}>
          <Typography fontSize={12} color={'#A4A9AC'} fontWeight={600} letterSpacing={-0.2}>
            {'대화 즐겨찾기'}
          </Typography>

          <ul>
            {bookmark.map((item, index) => (
              <li key={index} onClick={() => loadHistory(item.threadId)}>
                <Typography fontSize={14} fontWeight={500} fontFamily={'Pretendard'} letterSpacing={-0.2}>
                  {item.title}
                </Typography>
              </li>
            ))}
          </ul>

          <Divider variant="middle" sx={{ width: '100%', margin: '24px 0' }} />

          <Typography fontSize={12} color={'#A4A9AC'} fontWeight={600} letterSpacing={-0.2}>
            {'최근 대화'}
          </Typography>
          <ul>
            {searchResults.map((item, index) => (
              <li key={index} onClick={() => loadHistory(item.threadId)}>
                <Typography fontSize={14} fontWeight={500} fontFamily={'Pretendard'} letterSpacing={-0.2}>
                  {item.title}
                </Typography>
              </li>
            ))}
          </ul>
        </div>
      </Drawer> :

      <div className={sidebarIcon} onClick={handleDrawerToggle}>
        <SideBarIcon></SideBarIcon>
      </div>
    }
    </>
  )
}