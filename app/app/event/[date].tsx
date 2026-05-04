import { Stack, useLocalSearchParams } from 'expo-router';
import { useMemo } from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { AllocationBar } from '@/components/AllocationBar';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { findEvent, loadProfileDetail, summary } from '@/data/loadData';
import {
  formatCurrency,
  formatDateLong,
  formatQuarter,
  scaleFromBase,
} from '@/lib/format';
import { ProfileColors, SemanticColors } from '@/lib/palette';
import { useAppState } from '@/state/AppContext';

export default function EventDetailScreen() {
  const { date } = useLocalSearchParams<{ date: string }>();
  const { choice } = useAppState();

  const detail = useMemo(
    () => loadProfileDetail(choice.profileKey, choice.tiltEnabled),
    [choice.profileKey, choice.tiltEnabled]
  );
  const event = date ? findEvent(detail, date) : undefined;
  const profileColor = ProfileColors[choice.profileKey].primary;

  if (!event) {
    return (
      <ThemedView style={styles.root}>
        <SafeAreaView style={styles.flex}>
          <View style={styles.empty}>
            <ThemedText type="subtitle">No event on this date</ThemedText>
            <ThemedText style={styles.helper}>
              Quarterly rebalances happen on the first trading day of each quarter.
            </ThemedText>
          </View>
        </SafeAreaView>
      </ThemedView>
    );
  }

  const pre = event.portfolio_value_before_rebalance;
  const post = event.portfolio_value_after_rebalance;

  return (
    <ThemedView style={styles.root}>
      <Stack.Screen options={{ title: formatQuarter(event.date), headerBackTitle: 'Decisions' }} />
      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.headerSection}>
          <ThemedText style={[styles.eyebrow, { color: profileColor }]}>
            {detail.profile.name} · {choice.tiltEnabled ? 'AI tilt on' : 'no AI tilt'}
          </ThemedText>
          <ThemedText type="title">{formatQuarter(event.date)} rebalance</ThemedText>
          <ThemedText style={styles.helper}>{formatDateLong(event.date)}</ThemedText>
        </View>

        <Section title="What happened and why">
          <ThemedText style={styles.body}>{event.rationale_text}</ThemedText>
        </Section>

        {pre != null || post != null ? (
          <Section title="Portfolio value">
            <View style={styles.beforeAfter}>
              <ValueBlock label="Before" value={pre} capital={choice.capital} />
              <ValueBlock label="After (post-fees)" value={post} capital={choice.capital} />
            </View>
            <ThemedText style={styles.helper}>
              Difference is the transaction cost paid for this rebalance: $
              {event.transaction_cost.toFixed(2)} (per $1,000 base).
            </ThemedText>
          </Section>
        ) : null}

        <Section title="Target allocation after rebalance">
          <AllocationBar
            weights={event.asset_class_weights}
            labels={summary.asset_class_labels}
          />
        </Section>

        {event.tilt_applied && event.tilts.length > 0 ? (
          <Section title="AI tilt detail">
            <ThemedText style={styles.helper}>
              Each equity ticker started at equal weight inside the equity sleeve.
              The model nudged each by up to ±{summary.simulation.tilt_cap_pct}%.
              Tilts are zero-sum — the equity sleeve total stays at the profile target.
            </ThemedText>
            {event.tilts.map((t) => (
              <TiltRow key={t.ticker} ticker={t.ticker} tilt={t} />
            ))}
          </Section>
        ) : null}

        <Section title={`Trades placed (${event.trade_count})`}>
          <ThemedText style={styles.helper}>
            Dollar amounts are scaled to your selected capital ({formatCurrency(choice.capital)}).
            The simulation uses fractional shares so any starting amount works.
          </ThemedText>
          {event.trades.map((trade, idx) => (
            <TradeRow
              key={`${trade.ticker}-${idx}`}
              trade={{
                ...trade,
                shares: scaleFromBase(trade.shares, choice.capital),
                dollar_value: scaleFromBase(trade.dollar_value, choice.capital),
                transaction_cost: scaleFromBase(trade.transaction_cost, choice.capital),
              }}
            />
          ))}
        </Section>
      </ScrollView>
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

function ValueBlock({
  label,
  value,
  capital,
}: {
  label: string;
  value: number | null;
  capital: number;
}) {
  return (
    <View style={styles.valueBlock}>
      <ThemedText style={styles.valueLabel}>{label}</ThemedText>
      <ThemedText style={styles.valueAmount}>
        {value != null ? formatCurrency(scaleFromBase(value, capital)) : '—'}
      </ThemedText>
    </View>
  );
}

function TiltRow({
  ticker,
  tilt,
}: {
  ticker: string;
  tilt: import('@/data/types').Tilt;
}) {
  const color =
    tilt.direction === 'overweight'
      ? SemanticColors.positive
      : tilt.direction === 'underweight'
      ? SemanticColors.negative
      : SemanticColors.neutral;
  const sign = tilt.tilt > 0 ? '+' : '';
  return (
    <View style={styles.tiltRow}>
      <ThemedText style={styles.tiltTicker}>{ticker}</ThemedText>
      <View style={styles.tiltMid}>
        <ThemedText style={[styles.tiltDirection, { color }]}>{tilt.direction}</ThemedText>
        <ThemedText style={styles.tiltScore}>ML score {tilt.score.toFixed(4)}</ThemedText>
      </View>
      <ThemedText style={[styles.tiltAmount, { color }]}>
        {sign}
        {(tilt.tilt * 100).toFixed(1)}%
      </ThemedText>
    </View>
  );
}

function TradeRow({ trade }: { trade: import('@/data/types').Trade }) {
  const isBuy = trade.action === 'buy';
  const color = isBuy ? SemanticColors.positive : SemanticColors.negative;
  return (
    <View style={styles.tradeRow}>
      <View style={styles.tradeLeft}>
        <ThemedText style={[styles.tradeAction, { color }]}>{isBuy ? 'BUY' : 'SELL'}</ThemedText>
        <ThemedText style={styles.tradeTicker}>{trade.ticker}</ThemedText>
      </View>
      <View style={styles.tradeMid}>
        <ThemedText style={styles.tradeShares}>
          {trade.shares.toFixed(4)} sh
        </ThemedText>
        <ThemedText style={styles.tradePrice}>@ ${trade.price.toFixed(2)}</ThemedText>
      </View>
      <ThemedText style={styles.tradeValue}>{formatCurrency(trade.dollar_value)}</ThemedText>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  flex: { flex: 1 },
  scroll: { padding: 16, gap: 20, paddingBottom: 40 },
  empty: { padding: 24, gap: 8 },
  headerSection: { gap: 4 },
  eyebrow: { fontSize: 12, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.5 },
  helper: { fontSize: 12, opacity: 0.7, lineHeight: 18 },
  section: { gap: 10 },
  body: { fontSize: 14, lineHeight: 21 },
  beforeAfter: { flexDirection: 'row', gap: 12 },
  valueBlock: { flex: 1, gap: 2 },
  valueLabel: { fontSize: 11, opacity: 0.7, textTransform: 'uppercase' },
  valueAmount: { fontSize: 22, fontWeight: '700' },
  tiltRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#88888844',
    gap: 8,
  },
  tiltTicker: { fontSize: 14, fontWeight: '700', width: 56 },
  tiltMid: { flex: 1, gap: 2 },
  tiltDirection: { fontSize: 12, fontWeight: '600', textTransform: 'capitalize' },
  tiltScore: { fontSize: 11, opacity: 0.7 },
  tiltAmount: { fontSize: 16, fontWeight: '700' },
  tradeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#88888844',
    gap: 10,
  },
  tradeLeft: { flexDirection: 'row', alignItems: 'center', gap: 8, width: 90 },
  tradeAction: { fontSize: 11, fontWeight: '700' },
  tradeTicker: { fontSize: 14, fontWeight: '700' },
  tradeMid: { flex: 1, gap: 2 },
  tradeShares: { fontSize: 12 },
  tradePrice: { fontSize: 11, opacity: 0.7 },
  tradeValue: { fontSize: 14, fontWeight: '600' },
});
