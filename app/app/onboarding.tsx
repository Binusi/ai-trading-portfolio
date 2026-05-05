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
import { useAppState, type DepositDayOfMonth, type DepositPeriodMonths } from '@/state/AppContext';

const PERIOD_OPTIONS: { value: DepositPeriodMonths; label: string }[] = [
  { value: 1, label: 'Monthly' },
  { value: 2, label: 'Every 2 months' },
  { value: 3, label: 'Quarterly' },
  { value: 6, label: 'Twice a year' },
  { value: 12, label: 'Yearly' },
];

const DAY_OPTIONS: { value: DepositDayOfMonth; label: string }[] = [
  { value: 1, label: '1st' },
  { value: 15, label: '15th' },
  { value: 'EOM', label: 'End of month' },
];

export default function OnboardingScreen() {
  const { choice, setChoice } = useAppState();
  const [profileKey, setProfileKey] = useState<ProfileKey>(choice.profileKey);
  const [capitalText, setCapitalText] = useState(String(choice.capital));
  const [tiltEnabled, setTiltEnabled] = useState(choice.tiltEnabled);
  const [depositText, setDepositText] = useState(String(choice.deposit.amount || ''));
  const [periodMonths, setPeriodMonths] = useState<DepositPeriodMonths>(choice.deposit.periodMonths);
  const [dayOfMonth, setDayOfMonth] = useState<DepositDayOfMonth>(choice.deposit.dayOfMonth);
  const tint = useThemeColor({}, 'tint');
  const inputColor = useThemeColor({}, 'text');
  const inputBg = useThemeColor({}, 'icon');

  const capital = Math.max(1, Number(capitalText.replace(/[^0-9.]/g, '')) || 0);
  const depositAmount = Math.max(0, Number(depositText.replace(/[^0-9.]/g, '')) || 0);
  const depositActive = depositAmount > 0;
  const canStart = capital >= 1;

  const handleStart = async () => {
    await setChoice({
      profileKey,
      capital,
      tiltEnabled,
      deposit: {
        amount: depositAmount,
        periodMonths,
        dayOfMonth,
      },
    });
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

            <Section title="Regular contributions (optional)">
              <View style={[styles.inputWrap, { borderColor: inputBg + '66' }]}>
                <ThemedText style={styles.inputPrefix}>$</ThemedText>
                <TextInput
                  value={depositText}
                  onChangeText={setDepositText}
                  keyboardType="numeric"
                  placeholder="0"
                  placeholderTextColor={inputBg}
                  style={[styles.input, { color: inputColor }]}
                />
              </View>
              <ThemedText style={styles.subLabel}>How often</ThemedText>
              <SegmentedPicker
                options={PERIOD_OPTIONS}
                selected={periodMonths}
                onSelect={setPeriodMonths}
                disabled={!depositActive}
                tint={tint}
                inputBg={inputBg}
              />
              <ThemedText style={styles.subLabel}>Day of month</ThemedText>
              <SegmentedPicker
                options={DAY_OPTIONS}
                selected={dayOfMonth}
                onSelect={setDayOfMonth}
                disabled={!depositActive}
                tint={tint}
                inputBg={inputBg}
              />
              <ThemedText style={styles.helper}>
                Set $0 to skip and just invest the lump sum. Returns are computed
                by replaying the strategy&apos;s daily returns over your contribution
                schedule, so deposit-mode values may slightly overstate returns
                vs. a fresh simulation.
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

type SegmentedPickerProps<T extends string | number> = {
  options: { value: T; label: string }[];
  selected: T;
  onSelect: (value: T) => void;
  disabled?: boolean;
  tint: string;
  inputBg: string;
};

function SegmentedPicker<T extends string | number>({
  options, selected, onSelect, disabled, tint, inputBg,
}: SegmentedPickerProps<T>) {
  return (
    <View style={[styles.segmentRow, disabled && styles.segmentDisabled]}>
      {options.map((opt) => {
        const active = selected === opt.value;
        return (
          <Pressable
            key={String(opt.value)}
            onPress={() => !disabled && onSelect(opt.value)}
            style={({ pressed }) => [
              styles.segment,
              {
                borderColor: active ? tint : inputBg + '66',
                backgroundColor: active ? tint + '22' : 'transparent',
                opacity: disabled ? 0.4 : pressed ? 0.7 : 1,
              },
            ]}
          >
            <ThemedText style={[styles.segmentLabel, active && { color: tint, fontWeight: '600' }]}>
              {opt.label}
            </ThemedText>
          </Pressable>
        );
      })}
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
  subLabel: { fontSize: 12, opacity: 0.7, marginTop: 4 },
  segmentRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  segmentDisabled: { opacity: 0.6 },
  segment: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
    borderWidth: 1,
  },
  segmentLabel: { fontSize: 13 },
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
