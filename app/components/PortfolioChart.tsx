import { useMemo } from 'react';
import { Dimensions, StyleSheet, View } from 'react-native';
import { LineChart } from 'react-native-gifted-charts';

import { ThemedText } from '@/components/themed-text';
import type { BenchmarkDailyPoint } from '@/data/types';
import { scaleFromBase } from '@/lib/format';
import type { ReconstructedDay } from '@/lib/reconstructPortfolio';
import { useThemeColor } from '@/hooks/use-theme-color';

type Props = {
  reconstructed: ReconstructedDay[];     // user-scaled, deposit-aware series
  benchmark?: BenchmarkDailyPoint[];
  capital: number;                       // for SPY benchmark linear scaling only
  primaryColor: string;
  benchmarkColor?: string;
};

const SAMPLE_EVERY_N = 5; // weekly

export function PortfolioChart({
  reconstructed,
  benchmark,
  capital,
  primaryColor,
  benchmarkColor = '#888',
}: Props) {
  const text = useThemeColor({}, 'text');

  const { primaryData, benchmarkData, yMin, yMax } = useMemo(() => {
    const sampled = reconstructed.filter(
      (_, i) => i % SAMPLE_EVERY_N === 0 || i === reconstructed.length - 1
    );
    const primary = sampled.map((d) => ({
      value: d.value,
      label: '',
    }));

    let bench: { value: number; label: string }[] = [];
    if (benchmark && benchmark.length) {
      const benchSampled = benchmark.filter(
        (_, i) => i % SAMPLE_EVERY_N === 0 || i === benchmark.length - 1
      );
      bench = benchSampled.map((d) => ({
        value: scaleFromBase(d.value, capital),
        label: '',
      }));
    }

    const all = [...primary, ...bench];
    const min = Math.min(...all.map((p) => p.value));
    const max = Math.max(...all.map((p) => p.value));
    const padding = (max - min) * 0.05 || 1;
    return {
      primaryData: primary,
      benchmarkData: bench,
      yMin: Math.floor(min - padding),
      yMax: Math.ceil(max + padding),
    };
  }, [reconstructed, benchmark, capital]);

  const screenWidth = Dimensions.get('window').width;
  const chartWidth = screenWidth - 40;

  return (
    <View style={styles.wrapper}>
      <LineChart
        areaChart
        data={primaryData}
        data2={benchmarkData.length ? benchmarkData : undefined}
        color1={primaryColor}
        color2={benchmarkColor}
        startFillColor1={primaryColor}
        startOpacity={0.25}
        endOpacity={0.02}
        startFillColor2={benchmarkColor}
        thickness={2}
        thickness2={1.5}
        hideDataPoints
        yAxisColor={text + '33'}
        xAxisColor={text + '33'}
        xAxisLabelTextStyle={{ color: text, fontSize: 10 }}
        yAxisTextStyle={{ color: text, fontSize: 10 }}
        noOfSections={4}
        yAxisLabelPrefix="$"
        formatYLabel={(v) => formatYLabel(Number(v))}
        width={chartWidth}
        initialSpacing={4}
        spacing={Math.max(2, (chartWidth - 40) / Math.max(1, primaryData.length - 1))}
        adjustToWidth
        maxValue={yMax}
        mostNegativeValue={yMin}
        rulesType="solid"
        rulesColor={text + '11'}
      />
      <View style={styles.legend}>
        <Legend color={primaryColor} label="This portfolio" />
        {benchmarkData.length ? <Legend color={benchmarkColor} label="SPY buy & hold" /> : null}
      </View>
    </View>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <View style={styles.legendItem}>
      <View style={[styles.swatch, { backgroundColor: color }]} />
      <ThemedText style={styles.legendLabel}>{label}</ThemedText>
    </View>
  );
}

function formatYLabel(value: number): string {
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return value.toFixed(0);
}

const styles = StyleSheet.create({
  wrapper: { gap: 8 },
  legend: { flexDirection: 'row', gap: 16, paddingLeft: 8 },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  swatch: { width: 12, height: 12, borderRadius: 2 },
  legendLabel: { fontSize: 12, opacity: 0.85 },
});
