import { AccountOverview } from "@/components/pages/home/AccountOverview";
import ThemeChanger from "@/components/theme/theme-changer"

export default function Home() {
  return (
    <div>
      <div>Hello</div>
      <ThemeChanger />
      <AccountOverview />
    </div>
  );
}