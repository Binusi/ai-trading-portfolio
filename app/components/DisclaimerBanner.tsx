import { StyleSheet, View } from 'react-native';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { ThemedText } from '@/components/themed-text';

export function DisclaimerBanner({ text, compact = false }: { text: string; compact?: boolean }) {
  return (
    <View style={[styles.banner, compact && styles.compact]}>
      <IconSymbol name="exclamationmark.triangle" size={14} color="#8A6D00" />
      <ThemedText style={[styles.text, compact && styles.compactText]} lightColor="#5C4900" darkColor="#FFE08A">
        {text}
      </ThemedText>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    flexDirection: 'row',
    gap: 8,
    padding: 10,
    borderRadius: 8,
    backgroundColor: '#FFF4CC',
    alignItems: 'flex-start',
  },
  compact: { paddingVertical: 6 },
  text: { flex: 1, fontSize: 12, lineHeight: 16 },
  compactText: { fontSize: 11, lineHeight: 14 },
});
