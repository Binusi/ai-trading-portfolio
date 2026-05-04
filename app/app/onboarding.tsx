import { router } from 'expo-router';
import { useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { DisclaimerBanner } from '@/components/DisclaimerBanner';
import { ProfileOption } from '@/components/ProfileOption';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { summary } from '@/data/loadData';
import type { ProfileKey } from '@/data/types';
import { useThemeColor } from '@/hooks/use-theme-color';
import { ProfileColors } from '@/lib/palette';
import { useAppState } from '@/state/AppContext';

export default function OnboardingScreen() {
  const { choice, setChoice } = useAppState();
  const [profileKey, setProfileKey] = useState<ProfileKey>(choice.profileKey);
  const [capitalText, setCapitalText] = useState(String(choice.capital));
  const [tiltEnabled, setTiltEnabled] = useState(choice.tiltEnabled);
  const tint = useThemeColor({}, 'tint');
  const inputColor = useThemeColor({}, 'text');
  const inputBg = useThemeColor({}, 'icon');

  const capital = Math.max(1, Number(capitalText.replace(/[^0-9.]/g, '')) || 0);
  const canStart = capital >= 1;

  const handleStart = async () => {
    await setChoice({ profileKey, capital, tiltEnabled });
    router.replace('/(tabs)');
  };

  return (
    <ThemedView style={styles.root}>
      <SafeAreaView style={styles.flex} edges={['top']}>
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          style={styles.flex}
        >
          <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
            <ThemedText type="title">Set up your simulation</ThemedText>
            <ThemedText style={styles.subtitle}>
              Pick a risk profile, choose how much to invest, and we&apos;ll show you
              what would have happened over the last two years.
            </ThemedText>

            <DisclaimerBanner text={summary.disclaimer} />

            <Section title="Risk profile">
              <View style={styles.profileList}>
                {summary.profiles.map((p) => (
                  <ProfileOption
                    key={p.key}
                    profile={p}
                    selected={profileKey === p.key}
                    onSelect={setProfileKey}
                    assetClassLabels={summary.asset_class_labels}
                  />
                ))}
              </View>
            </Section>

            <Section title="Starting capital">
              <View style={[styles.inputWrap, { borderColor: inputBg + '66' }]}>
                <ThemedText style={styles.inputPrefix}>$</ThemedText>
                <TextInput
                  value={capitalText}
                  onChangeText={setCapitalText}
                  keyboardType="numeric"
                  placeholder="1000"
                  placeholderTextColor={inputBg}
                  style={[styles.input, { color: inputColor }]}
                />
              </View>
              <ThemedText style={styles.helper}>
                Returns scale linearly. Try $100, $1,000, or $100,000 — the percentage
                outcome is the same; the dollar values just stretch.
              </ThemedText>
            </Section>

            <Section title="AI tilt (optional)">
              <View style={styles.toggleRow}>
                <View style={styles.toggleText}>
                  <ThemedText type="defaultSemiBold">Apply AI tilt to equity sleeve</ThemedText>
                  <ThemedText style={styles.helper}>
                    Tilts individual stock weights by ±5% based on a machine-learning
                    signal. Backtest result: small upside, similar drawdown.
                    Honest take: probably a wash.
                  </ThemedText>
                </View>
                <Switch
                  value={tiltEnabled}
                  onValueChange={setTiltEnabled}
                  trackColor={{ true: tint }}
                />
              </View>
            </Section>

            <Pressable
              onPress={handleStart}
              disabled={!canStart}
              style={({ pressed }) => [
                styles.cta,
                {
                  backgroundColor: ProfileColors[profileKey].primary,
                  opacity: pressed ? 0.85 : canStart ? 1 : 0.4,
                },
              ]}
            >
              <ThemedText style={styles.ctaText} lightColor="#fff" darkColor="#fff">
                Run the simulation
              </ThemedText>
            </Pressable>
          </ScrollView>
        </KeyboardAvoidingView>
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

const styles = StyleSheet.create({
  root: { flex: 1 },
  flex: { flex: 1 },
  scroll: { padding: 16, gap: 20, paddingBottom: 40 },
  subtitle: { fontSize: 14, opacity: 0.8, lineHeight: 20 },
  section: { gap: 10 },
  profileList: { gap: 10 },
  inputWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 4,
  },
  inputPrefix: { fontSize: 18, opacity: 0.6, marginRight: 4 },
  input: { flex: 1, fontSize: 18, paddingVertical: 10 },
  helper: { fontSize: 12, opacity: 0.7, lineHeight: 18 },
  toggleRow: { flexDirection: 'row', alignItems: 'center', gap: 16 },
  toggleText: { flex: 1, gap: 4 },
  cta: {
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 4,
  },
  ctaText: { fontSize: 16, fontWeight: '700' },
});
