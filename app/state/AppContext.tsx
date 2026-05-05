// Holds the user's onboarding choices: profile, capital, AI tilt opt-in,
// optional periodic-deposit schedule. Persisted to AsyncStorage so reopening
// the app skips onboarding.

import AsyncStorage from '@react-native-async-storage/async-storage';
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';

import { summary } from '@/data/loadData';
import type { ProfileKey } from '@/data/types';

export type DepositPeriodMonths = 1 | 2 | 3 | 6 | 12;
export type DepositDayOfMonth = 1 | 15 | 'EOM';

export type DepositSchedule = {
  amount: number;                 // 0 = deposits disabled
  periodMonths: DepositPeriodMonths;
  dayOfMonth: DepositDayOfMonth;
};

type UserChoice = {
  profileKey: ProfileKey;
  capital: number;
  tiltEnabled: boolean;
  deposit: DepositSchedule;
};

type AppState = {
  ready: boolean;
  onboarded: boolean;
  choice: UserChoice;
  setChoice: (next: UserChoice) => Promise<void>;
  resetOnboarding: () => Promise<void>;
};

const STORAGE_KEY = 'app_user_choice_v2';
const ONBOARDED_KEY = 'app_onboarded_v1';

const defaultDeposit: DepositSchedule = {
  amount: 0,
  periodMonths: 1,
  dayOfMonth: 1,
};

const defaultChoice: UserChoice = {
  profileKey: summary.default_view.profile_key,
  capital: 1000,
  tiltEnabled: summary.default_view.tilt_enabled,
  deposit: defaultDeposit,
};

const AppContext = createContext<AppState | null>(null);

function normalizeDeposit(raw: unknown): DepositSchedule {
  if (!raw || typeof raw !== 'object') return defaultDeposit;
  const r = raw as Partial<DepositSchedule>;
  const amount = typeof r.amount === 'number' && r.amount >= 0 ? r.amount : 0;
  const periodMonths: DepositPeriodMonths =
    r.periodMonths === 1 || r.periodMonths === 2 || r.periodMonths === 3 ||
    r.periodMonths === 6 || r.periodMonths === 12
      ? r.periodMonths
      : 1;
  const dayOfMonth: DepositDayOfMonth =
    r.dayOfMonth === 15 || r.dayOfMonth === 'EOM' ? r.dayOfMonth : 1;
  return { amount, periodMonths, dayOfMonth };
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [onboarded, setOnboarded] = useState(false);
  const [choice, setChoiceState] = useState<UserChoice>(defaultChoice);

  useEffect(() => {
    (async () => {
      try {
        const [storedChoice, storedOnboarded] = await Promise.all([
          AsyncStorage.getItem(STORAGE_KEY),
          AsyncStorage.getItem(ONBOARDED_KEY),
        ]);
        if (storedChoice) {
          const parsed = JSON.parse(storedChoice) as Partial<UserChoice>;
          setChoiceState({
            profileKey: parsed.profileKey ?? defaultChoice.profileKey,
            capital: typeof parsed.capital === 'number' ? parsed.capital : defaultChoice.capital,
            tiltEnabled: parsed.tiltEnabled ?? defaultChoice.tiltEnabled,
            deposit: normalizeDeposit(parsed.deposit),
          });
        }
        if (storedOnboarded === 'true') setOnboarded(true);
      } finally {
        setReady(true);
      }
    })();
  }, []);

  const setChoice = async (next: UserChoice) => {
    setChoiceState(next);
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    await AsyncStorage.setItem(ONBOARDED_KEY, 'true');
    setOnboarded(true);
  };

  const resetOnboarding = async () => {
    await AsyncStorage.removeItem(ONBOARDED_KEY);
    setOnboarded(false);
  };

  const value = useMemo<AppState>(
    () => ({ ready, onboarded, choice, setChoice, resetOnboarding }),
    [ready, onboarded, choice]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useAppState(): AppState {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppState must be used inside AppProvider');
  return ctx;
}
