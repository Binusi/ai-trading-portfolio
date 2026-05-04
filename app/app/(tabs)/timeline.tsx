import { Redirect, router } from 'expo-router';
import { useMemo } from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { DisclaimerBanner } from '@/components/DisclaimerBanner';
import { RebalanceEventRow } from '@/components/RebalanceEventRow';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { loadProfileDetail, summary } from '@/data/loadData';
import { useAppState } from '@/state/AppContext';

export default function TimelineScreen() {
  const { ready, onboarded, choice } = useAppState();

  const detail = useMemo(
    () => loadProfileDetail(choice.profileKey, choice.tiltEnabled),
    [choice.profileKey, choice.tiltEnabled]
  );

  if (!ready) return <ThemedView style={styles.flex} />;
  if (!onboarded) return <Redirect href="/onboarding" />;

  return (
    <ThemedView style={styles.root}>
      <SafeAreaView style={styles.flex} edges={['top']}>
        <ScrollView>
          <View style={styles.header}>
            <ThemedText type="title">Decisions</ThemedText>
            <ThemedText style={styles.subtitle}>
              Each row is a quarterly rebalance. Tap to see the rationale, the
              specific trades, and the allocation snapshot for that day.
            </ThemedText>
            <DisclaimerBanner text={summary.disclaimer} compact />
          </View>

          {detail.rebalance_events.map((event) => (
            <RebalanceEventRow
              key={event.date}
              event={event}
              capital={choice.capital}
              profileKey={choice.profileKey}
              onPress={() => router.push(`/event/${event.date}`)}
            />
          ))}
        </ScrollView>
      </SafeAreaView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  flex: { flex: 1 },
  header: { padding: 16, gap: 8 },
  subtitle: { fontSize: 13, opacity: 0.8, lineHeight: 18 },
});
