import { router } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Switch, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { DisclaimerBanner } from '@/components/DisclaimerBanner';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { summary } from '@/data/loadData';
import { useAppState } from '@/state/AppContext';
import { useThemeColor } from '@/hooks/use-theme-color';
import { ProfileColors } from '@/lib/palette';

export default function AboutScreen() {
  const { choice, setChoice } = useAppState();
  const tint = useThemeColor({}, 'tint');

  const toggleTilt = async (next: boolean) => {
    await setChoice({ ...choice, tiltEnabled: next });
  };

  return (
    <ThemedView style={styles.root}>
      <SafeAreaView style={styles.flex} edges={['top']}>
        <ScrollView contentContainerStyle={styles.scroll}>
          <ThemedText type="title">About</ThemedText>

          <Section title="What this app does">
            <ThemedText style={styles.body}>
              This is a simulator. It replays the last two years of market data
              against three risk profiles (Conservative / Balanced / Aggressive),
              rebalancing quarterly. Each profile holds a fixed mix of US stocks,
              bonds, and ETFs. An optional AI tilt nudges individual stock weights
              by up to ±5% based on a machine-learning return signal.
            </ThemedText>
            <ThemedText style={styles.body}>
              The dashboard shows what would have happened to{' '}
              <ThemedText type="defaultSemiBold">your</ThemedText> entered capital
              under that strategy, scaled linearly from the $1,000 base simulation.
            </ThemedText>
          </Section>

          <Section title="Active simulation">
            <Row label="Profile" value={choice.profileKey} />
            <Row label="Capital" value={`$${choice.capital.toLocaleString()}`} />
            <View style={styles.switchRow}>
              <ThemedText style={styles.label}>AI tilt enabled</ThemedText>
              <Switch
                value={choice.tiltEnabled}
                onValueChange={toggleTilt}
                trackColor={{ true: tint }}
              />
            </View>
            <Pressable
              onPress={() => router.push('/onboarding')}
              style={({ pressed }) => [
                styles.button,
                { borderColor: ProfileColors[choice.profileKey].primary },
                pressed && { opacity: 0.7 },
              ]}
            >
              <ThemedText
                style={styles.buttonText}
                lightColor={ProfileColors[choice.profileKey].primary}
                darkColor={ProfileColors[choice.profileKey].primary}
              >
                Change profile or capital
              </ThemedText>
            </Pressable>
          </Section>

          <Section title="Underlying ML model">
            <Row label="Algorithm" value={summary.ml_model.model_name} />
            <Row label="Target" value={summary.ml_model.target} />
            <Row
              label="Validation Sharpe"
              value={
                summary.ml_model.validation_sharpe != null
                  ? summary.ml_model.validation_sharpe.toFixed(2)
                  : '—'
              }
            />
            <ThemedText style={styles.body}>
              The ML signal only drives the optional AI tilt. The asset-class
              allocation in each profile is fixed and does not depend on the
              model.
            </ThemedText>
          </Section>

          <Section title="Simulation parameters">
            <Row label="Period" value={`${summary.simulation.start_date} → ${summary.simulation.end_date}`} />
            <Row label="Rebalance" value={summary.simulation.rebalance_cadence} />
            <Row label="Tilt cap" value={`±${summary.simulation.tilt_cap_pct}%`} />
            <Row label="Trading cost" value={`${summary.simulation.transaction_cost_bps} bps per trade`} />
            <Row label="Universe" value={summary.simulation.simulation_tickers.join(', ')} />
            <Row label="Data generated" value={summary.generated_at.split('T')[0]} />
          </Section>

          <DisclaimerBanner text={summary.disclaimer} />
        </ScrollView>
      </SafeAreaView>
    </ThemedView>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <View style={styles.section}>
      <ThemedText type="subtitle">{title}</ThemedText>
      {children}
    </View>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.row}>
      <ThemedText style={styles.label}>{label}</ThemedText>
      <ThemedText style={styles.rowValue}>{value}</ThemedText>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  flex: { flex: 1 },
  scroll: { padding: 16, gap: 20, paddingBottom: 40 },
  section: { gap: 8 },
  body: { fontSize: 13, lineHeight: 19, opacity: 0.9 },
  row: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4, gap: 12 },
  label: { fontSize: 13, opacity: 0.7 },
  rowValue: { fontSize: 13, fontWeight: '600', flex: 1, textAlign: 'right' },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 6,
  },
  button: {
    borderWidth: 1,
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: 'center',
    marginTop: 6,
  },
  buttonText: { fontSize: 14, fontWeight: '600' },
});
