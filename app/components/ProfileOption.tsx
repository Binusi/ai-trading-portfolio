import { Pressable, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import type { ProfileKey, ProfileMeta } from '@/data/types';
import { ProfileColors } from '@/lib/palette';
import { formatPercent } from '@/lib/format';

type Props = {
  profile: ProfileMeta;
  selected: boolean;
  onSelect: (key: ProfileKey) => void;
  assetClassLabels: Record<string, string>;
};

export function ProfileOption({ profile, selected, onSelect, assetClassLabels }: Props) {
  const colors = ProfileColors[profile.key];
  const orderedTargets = Object.entries(profile.targets)
    .filter(([, v]) => v > 0)
    .sort(([, a], [, b]) => b - a);

  return (
    <Pressable
      onPress={() => onSelect(profile.key)}
      style={({ pressed }) => [
        styles.card,
        {
          borderColor: selected ? colors.primary : '#88888833',
          backgroundColor: selected ? colors.soft : 'transparent',
        },
        pressed && { opacity: 0.85 },
      ]}
    >
      <View style={styles.header}>
        <View style={[styles.dot, { backgroundColor: colors.primary }]} />
        <ThemedText
          type="subtitle"
          lightColor={selected ? colors.primary : undefined}
          darkColor={selected ? colors.primary : undefined}
        >
          {profile.name}
        </ThemedText>
      </View>
      <ThemedText style={styles.description}>{profile.description}</ThemedText>
      <View style={styles.targetsRow}>
        {orderedTargets.map(([cls, w]) => (
          <ThemedText key={cls} style={styles.target}>
            {formatPercent(w, 0)} {assetClassLabels[cls] ?? cls}
          </ThemedText>
        ))}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 12,
    borderWidth: 2,
    padding: 14,
    gap: 8,
  },
  header: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  dot: { width: 10, height: 10, borderRadius: 5 },
  description: { fontSize: 13, lineHeight: 18, opacity: 0.85 },
  targetsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  target: {
    fontSize: 11,
    paddingHorizontal: 6,
    paddingVertical: 3,
    borderRadius: 4,
    backgroundColor: '#88888822',
    overflow: 'hidden',
  },
});
