import { StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { useThemeColor } from '@/hooks/use-theme-color';

type Props = {
  label: string;
  value: string;
  caption?: string;
  valueColor?: string;
};

export function MetricCard({ label, value, caption, valueColor }: Props) {
  const border = useThemeColor({}, 'icon');
  return (
    <ThemedView style={[styles.card, { borderColor: border + '33' }]}>
      <ThemedText style={styles.label}>{label}</ThemedText>
      <ThemedText style={[styles.value, valueColor ? { color: valueColor } : undefined]}>
        {value}
      </ThemedText>
      {caption ? <ThemedText style={styles.caption}>{caption}</ThemedText> : null}
    </ThemedView>
  );
}

export function MetricGrid({ children }: { children: React.ReactNode }) {
  return <View style={styles.grid}>{children}</View>;
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    minWidth: '45%',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
    gap: 4,
  },
  label: { fontSize: 12, opacity: 0.7 },
  value: { fontSize: 20, fontWeight: '700' },
  caption: { fontSize: 11, opacity: 0.6 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
});
