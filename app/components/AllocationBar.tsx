import { StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import type { AssetClass } from '@/data/types';
import { AssetClassColors } from '@/lib/palette';
import { formatPercent } from '@/lib/format';

type Props = {
  weights: Partial<Record<AssetClass, number>>;
  labels?: Record<string, string>;
  showLegend?: boolean;
  height?: number;
};

const ORDER: AssetClass[] = [
  'us_equity',
  'equity_index_etf',
  'international_etf',
  'commodity_etf',
  'bond_etf',
];

export function AllocationBar({ weights, labels, showLegend = true, height = 14 }: Props) {
  const entries = ORDER.filter((k) => (weights[k] ?? 0) > 0).map((k) => ({
    key: k,
    weight: weights[k]!,
    color: AssetClassColors[k],
    label: labels?.[k] ?? k,
  }));

  return (
    <View style={styles.container}>
      <View style={[styles.bar, { height }]}>
        {entries.map((e) => (
          <View
            key={e.key}
            style={{ flex: e.weight, backgroundColor: e.color }}
          />
        ))}
      </View>
      {showLegend ? (
        <View style={styles.legend}>
          {entries.map((e) => (
            <View key={e.key} style={styles.legendItem}>
              <View style={[styles.swatch, { backgroundColor: e.color }]} />
              <ThemedText style={styles.legendLabel}>
                {e.label} {formatPercent(e.weight, 1)}
              </ThemedText>
            </View>
          ))}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { gap: 8 },
  bar: {
    flexDirection: 'row',
    overflow: 'hidden',
    borderRadius: 4,
    width: '100%',
  },
  legend: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  swatch: { width: 10, height: 10, borderRadius: 2 },
  legendLabel: { fontSize: 11, opacity: 0.85 },
});
