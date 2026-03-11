import { Options, TooltipFormatterCallbackFunction } from 'highcharts';

const CommonChartOptions: Options = { // 공통 차트 옵션
  credits: {
    enabled: false
  },
  chart: {
    backgroundColor: 'transparent',
    style: {
      fontFamily: 'Pretendard',
    }
  },
  colors: [
    "#8578F8",
    "#666E80",
    "#6EDDCC",
    "#77B1ED",
    "#A0A1B8",
    "#C69DFF",
    "#5469A0",
    "#F081C4"
  ],
  accessibility: {
    enabled: false
  },
  legend: {
    itemStyle: {
      fontSize: '14px',
      fontWeight: 'normal',
      // fontFamily: 'Pretendard, sans-serif',
    }
  }
}

const DefaultBarChart = (config: Options) => { // 기본 Bar 디자인 옵션
  const BarOption: Options = {
    chart: {
      ...config.chart,
      type: 'column'
    },

    series: [
      ...(config.series ?? [])
    ],

    title: {
      ...config.title,
      text: '',
    },

    xAxis: {
      ...(config.xAxis ?? {}),
      labels: {
        style: {
          fontSize: '14px'
        }
      }
    },

    yAxis: {
      ...(config.yAxis ?? {}),
      title: {
        text: ''
      }
    },

    tooltip: {
      useHTML: true,
      backgroundColor: 'transparent',
      borderRadius: 4,
      borderColor: '#E5E7EB',
      shadow: true,
      padding: 0,
      formatter: function () {
        const v = this.y as number;

        const dept = this.series.name;
        const color = this.color;

        // ✅ HTML 커스텀 템플릿
        return `
          <div style="
            border-radius: 6px;
            overflow: hidden;
            font-family: Pretendard, sans-serif;
            border: 1px solid #e5e7eb;
          ">
            <div style="
              background-color: #f3f4f6;
              padding: 6px 10px;
              font-size: 13px;
              font-weight: 600;
              color: #111827;
              border-bottom: 1px solid #e5e7eb;
            ">
              ${dept}
            </div>
            <div style="
              background-color: #fff;
              padding: 8px 12px;
              display: flex;
              align-items: center;
              gap: 6px;
            ">
              <div style="
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: ${color};
              "></div>
              <div style="font-size: 13px; color: #111827;">
                <strong>${v.toLocaleString('ko-KR', { minimumFractionDigits: 0, maximumFractionDigits: 2 })}</strong>
              </div>
            </div>
          </div>
        `;
      } as TooltipFormatterCallbackFunction,
    },

    legend: {
      ...config.legend,
      itemMarginTop: 15,
    },
    plotOptions: {
      ...config.plotOptions,
      column: {
        maxPointWidth: 96
      }
    },
  }

  return BarOption;
}

const DefaultPieChart = (config: Options) => { // 기본 Pie 디자인 옵션
  const PieOption: Options = {
    ...config,
    chart: {
      ...config.chart,
      type: 'pie',
      marginBottom: 70,
    },
    title: {
      ...config.title,
      text: '',
    },
    legend: {
      ...config.legend,
      enabled: true,
      layout: 'horizontal',
      align: 'center',
      verticalAlign: 'bottom',
      margin: 0,
    },
    tooltip: {
      ...config.tooltip,
      enabled: true,
      pointFormat:
        "<span style='color:{point.color}'>●</span> {point.name}: {point.percentage:.1f}%",
    },

    plotOptions: {
      ...config.plotOptions,
      pie: {
        innerSize: '60%',
        borderRadius: 0,
        borderWidth: 0,
        dataLabels: {
          enabled: true,
          format: '{point.percentage:.1f}%',
          style: {
            fontSize: '14px',
            fontWeight: 'bold',
            color: '#ffffff',
            textOutline: 'none',
            textAlign: 'none',
          },
          distance: -30,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
        },
        showInLegend: true,
      },
    },
  }

  return PieOption;
}

const DefaultLineChart = (config: Options) => { // 기본 Line 디자인 옵션
  const LineOption: Options = {
    ...config,

    chart: {
      ...config.chart,
      type: 'spline',
    },
    title: {
      ...config.title,
      text: '',
    },
    xAxis: {
      ...(config.xAxis ?? {}),
      labels: {
        style: {
          fontSize: '14px'
        }
      }
    },
    yAxis: {
      ...(config.yAxis ?? {}),
      title: {
        text: ''
      }
    },
    tooltip: {
      useHTML: true,
      backgroundColor: 'transparent',
      borderRadius: 4,
      borderColor: '#E5E7EB',
      shadow: true,
      padding: 0,
      formatter: function () {
        const v = this.y as number;

        const dept = this.series.name;
        const color = this.color;

        // ✅ HTML 커스텀 템플릿
        return `
          <div style="
            border-radius: 6px;
            overflow: hidden;
            font-family: Pretendard, sans-serif;
            border: 1px solid #e5e7eb;
          ">
            <div style="
              background-color: #f3f4f6;
              padding: 6px 10px;
              font-size: 13px;
              font-weight: 600;
              color: #111827;
              border-bottom: 1px solid #e5e7eb;
            ">
              ${dept}
            </div>
            <div style="
              background-color: #fff;
              padding: 8px 12px;
              display: flex;
              align-items: center;
              gap: 6px;
            ">
              <div style="
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: ${color};
              "></div>
              <div style="font-size: 13px; color: #111827;">
                <strong>${v.toLocaleString('ko-KR', { minimumFractionDigits: 0, maximumFractionDigits: 2 })}</strong>
              </div>
            </div>
          </div>
        `;
      } as TooltipFormatterCallbackFunction,
    },
    legend: {
      ...config.legend,
    },
    plotOptions: {
      ...config.plotOptions
    },
  }
  return LineOption;
}

export {
  CommonChartOptions,
  DefaultBarChart,
  DefaultPieChart,
  DefaultLineChart
}