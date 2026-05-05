import { Pressable, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import type { ProfileKey, RebalanceEvent } from '@/data/types';
import { ProfileColors, SemanticColors } from '@/lib/palette';
import { formatCurrency, formatDate, formatQuarter } from '@/lib/format';

type Props = {
  event: RebalanceEvent;
  /** Pre-scaled portfolio value for this event date (deposit-aware). */
  postValueScaled: number | null;
  profileKey: ProfileKey;
  onPress: () => void;
};

export function RebalanceEventRow({ event, postValueScaled, profileKey, onPress }: Props) {
  const profileColor = ProfileColors[profileKey].primary;

  return (
    <Pressable onPress={onPress} style={({ pressed }) => [styles.row, pressed && { opacity: 0.7 }]}>
      <View style={[styles.timelineDot, { backgroundColor: profileColor }]} />
      <View style={styles.body}>
        <View style={styles.headerRow}>
          <ThemedText style={styles.quarter}>{formatQuarter(event.date)}</ThemedText>
          <ThemedText style={styles.date}>{formatDate(event.date)}</ThemedText>
        </View>
        <View style={styles.metricsRow}>
          <ThemedText style={styles.metric}>
            {formatCurrency(postValueScaled ?? 0, { compact: true })} after
          </ThemedText>
          <ThemedText style={styles.metric}>
            {event.trade_count} trade{event.trade_count === 1 ? '' : 's'}
          </ThemedText>
          <ThemedText
            style={[
              styles.metric,
              event.tilt_applied
                ? { color: SemanticColors.positive }
                : { color: SemanticColors.neutral },
            ]}
          >
            {event.tilt_applied ? 'AI tilt applied' : 'profile only'}
          </ThemedText>
        </View>
      </View>
      <ThemedText style={styles.chevron}>›</ThemedText>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#88888844',
  },
  timelineDot: { width: 10, height: 10, borderRadius: 5 },
  body: { flex: 1, gap: 4 },
  headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' },
  quarter: { fontSize: 16, fontWeight: '700' },
  date: { fontSize: 12, opacity: 0.6 },
  metricsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  metric: { fontSize: 12, opacity: 0.85 },
  chevron: { fontSize: 24, opacity: 0.4 },
});
