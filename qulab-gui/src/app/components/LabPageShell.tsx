import { type ReactNode } from 'react';
import { EchoLabCommandInline } from './EchoLabCommandInline';
import { Navigation } from './Navigation';

/** Standard lab layout: nav + visible Echo input + page content. */
export function LabPageShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-24 px-[32px] md:px-[48px]">
        <div className="max-w-[1440px] mx-auto">
          <EchoLabCommandInline className="mb-8" />
          {children}
        </div>
      </main>
    </div>
  );
}
