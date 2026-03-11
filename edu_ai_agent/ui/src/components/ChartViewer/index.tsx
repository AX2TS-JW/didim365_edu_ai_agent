import type { Options } from 'highcharts';
import { useEffect, useState } from "react";
import style from './ChatViewer.module.scss';
import { Typography } from "@mui/material";
import { ArrowDownIcon } from "../Icons";
import HighchartsReact from 'highcharts-react-official';
import Highcharts from 'highcharts';
import { CommonChartOptions, DefaultBarChart, DefaultLineChart, DefaultPieChart } from '@/types/graphVM';

type ChartType = 'column' | 'pie' | 'line';

interface ChartViewerProps {
  data: any; // 수정 필요
  collapse: boolean,
}

export const ChartViewer = (props: ChartViewerProps) => {
  const { data, collapse } = props;
  const [isCollapsed, setIsCollapsed] = useState(collapse);
  const [options, setOptions] = useState<Options>()
  const { container, header, title, icon, chartArea } = style;

  const handleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  }

  const createChartOptions = (chartType: string) => {
    let chartOptions: Options;

    switch (chartType) {
      case 'column':
        chartOptions = DefaultBarChart(data);
        break;
      case 'pie':
        chartOptions = DefaultPieChart(data);
        break;
      case 'line':
        chartOptions = DefaultLineChart(data);
        break;
      default:
        console.log('unKnown chart Type')
        return null;
    };

    return {
      ...CommonChartOptions,
      ...chartOptions,
      chart: {
        ...CommonChartOptions.chart,
        ...chartOptions.chart
      },
      legend: {
        ...CommonChartOptions.legend,
        ...chartOptions.legend
      }
    }
  };

  useEffect(() => {
    const chartType = data?.chart?.type as ChartType;

    if (!chartType) return;
    const chartOptions = createChartOptions(chartType);
    if (chartOptions) {
      setOptions(chartOptions);
    }
  }, [data]);

  return (
    <div className={container}>
      <div className={header} onClick={handleCollapse}>
        <div className={title}>
          <Typography variant='h6'>
            {'차트'}
          </Typography>
          <ArrowDownIcon></ArrowDownIcon>
        </div>
      </div>

      {isCollapsed && options && (
        <div style={{ margin: '0px 20px 16px 20px' }} className={chartArea}>
          <HighchartsReact highcharts={Highcharts} options={options} />
        </div>
      )}
    </div>
  )
}